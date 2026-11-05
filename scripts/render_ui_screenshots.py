"""Render the web-interface figures of the thesis (Figures 4 and 5) from a SAVED pipeline run.

The real React frontend (frontend/dist, built with `npm run build`) is served together with a
tiny stand-in for the FastAPI backend that answers POST /api/analyze with the outputs saved in
results/case_study_rerun/ (final system, 5 Nov 2026) instead of running the models again. The
detection overlay is drawn with the backend's own helper (backend/services/visualization.py).
So the screenshots show exactly the saved run, and anyone can reproduce them without the
models, the textbooks or an OpenAI key.

Usage (repo root; needs `pip install playwright` and a local Chrome/Chromium):
    cd frontend && npm ci && npm run build && cd ..
    python scripts/render_ui_screenshots.py --chrome "C:/Program Files/Google/Chrome/Application/chrome.exe"
Output: thesis/figures/0{1,2,3,4}_*.png and the same files in figures/webapp/.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))


def build_response(run: Path, image: Path) -> dict:
    from PIL import Image
    from services.visualization import render_overlay_base64  # backend helper

    s1 = json.loads((run / "stage1_detection.json").read_text(encoding="utf-8"))
    s2 = json.loads((run / "stage2_classification.json").read_text(encoding="utf-8"))
    s3 = json.loads((run / "stage3_reasoning.json").read_text(encoding="utf-8"))
    with Image.open(image) as src:
        overlay = render_overlay_base64(src, s1["per_image"][0]["boxes"])
    meta_file = run / "metadata.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    meta["note"] = f"rendered from the saved run in {run.relative_to(ROOT)}"
    return {
        "metadata": meta,
        "stage1_detection": s1, "stage2_classification": s2, "stage3_reasoning": s3,
        "annotated_image_base64": overlay,
    }


def make_handler(dist: Path, samples: Path, payload: bytes):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(dist), **kw)

        def log_message(self, *args):  # quiet
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode() if not isinstance(obj, bytes) else obj
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/health":
                return self._json({"status": "ok", "pipeline_ready": True, "version": "1.0.0"})
            if self.path == "/api/samples":
                files = sorted(p for p in samples.iterdir() if p.suffix.lower() in {".jpg", ".png"})
                return self._json({"samples": [{"name": p.name, "size_bytes": p.stat().st_size} for p in files]})
            if self.path.startswith("/api/samples/"):
                p = samples / Path(self.path.split("/")[-1]).name
                data = p.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            return super().do_GET()

        def do_POST(self):
            if self.path == "/api/analyze":
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                return self._json(payload)
            self._json({"detail": "not found"}, 404)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=ROOT / "results/case_study_rerun")
    ap.add_argument("--image", default="BCCD_00010.jpg")
    ap.add_argument("--dist", type=Path, default=ROOT / "frontend/dist")
    ap.add_argument("--chrome", default=None, help="path to a Chrome/Chromium executable")
    ap.add_argument("--out", type=Path, default=ROOT / "thesis/figures")
    args = ap.parse_args()
    samples = ROOT / "examples/sample_images"
    payload = json.dumps(build_response(args.run, samples / args.image)).encode()

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(args.dist, samples, payload))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/"

    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=args.chrome) if args.chrome else pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1240, "height": 1000}, device_scale_factor=2)
        page.goto(url)
        page.get_by_role("button", name=args.image).click()
        page.get_by_text("Agent Trace (").wait_for(timeout=15000)
        page.wait_for_timeout(800)

        def card(heading: str):
            return page.locator(".card").filter(has=page.get_by_role("heading", name=heading, exact=False)).first

        def union(*locs):
            boxes = [l.bounding_box() for l in locs]
            x0 = min(b["x"] for b in boxes); y0 = min(b["y"] for b in boxes)
            x1 = max(b["x"] + b["width"] for b in boxes); y1 = max(b["y"] + b["height"] for b in boxes)
            return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}

        args.out.mkdir(parents=True, exist_ok=True)
        shots = {
            "01_detection.png": union(card("Stage 1")),
            "02_classification_gradcam.png": union(card("Stage 2"), card("Grad-CAM")),
            "03_reasoning.png": union(card("Stage 3")),
            "04_agent_trace.png": union(card("Agent Trace")),
        }
        for name, clip in shots.items():
            page.screenshot(path=str(args.out / name), clip=clip, full_page=True)
            print("saved", args.out / name, f"{clip['width']:.0f}x{clip['height']:.0f}")
        browser.close()
    server.shutdown()
    webapp = ROOT / "figures/webapp"
    if webapp.is_dir() and args.out.resolve() != webapp.resolve():
        for name in shots:
            shutil.copy2(args.out / name, webapp / name)


if __name__ == "__main__":
    main()
