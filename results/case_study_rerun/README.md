# Case study (thesis Section 6.6): evidence

Image: bundled sample `examples/sample_images/BCCD_00010.jpg` (640x480, from the BCCD collection).

`pipeline.save_intermediate_results: true` makes every run overwrite `results/stage*.json`, and
`logs/` is not tracked by git, so all evidence for the case study is kept in this folder.

| File(s) | Run | System | Result |
|---|---|---|---|
| `original_run_20260428_145040.log` | 28 Apr 2026, 14:50 | v1 classifier, no minimum-cell rule | 2 WBC, 17 RBC, 0 flagged, review **False**. The run quoted in thesis Appendix C (same agent queries and safety-flag text). |
| `original_run_20260428_140721.log` | 28 Apr 2026, 14:07 | same configuration, run from an older copy of the project | 2 WBC, 17 RBC, 0 flagged, review **False** (same failure) |
| `rerun_20260929_150829.log` | 29 Sep 2026 | v2 classifier + minimum-cell rule | 1 flagged (ig 0.42, HIGH), review **True**; its output files were later overwritten |
| `before_citation_check_20261105_2002/` | 5 Nov 2026, 20:02 (web UI) | v2 + minimum-cell rule + BGE-small embedder | review **True** (`HIGH_UNCERTAINTY`, `INSUFFICIENT_CELL_COUNT`). The agent never called the knowledge-base tool but cited [Reference 1] and [Reference 2], and the pipeline showed its own pre-fetched chunks as "retrieved references". This led to the citation check. |
| `stage1_detection.json`, `stage2_classification.json`, `stage3_reasoning.json`, `metadata.json`, `pipeline_log.txt` | 5 Nov 2026, 21:12 (CLI: `python main.py analyze examples/sample_images/BCCD_00010.jpg`) | **final system** (as above + agent citation check) | ig 0.423 / entropy 1.228 / HIGH, monocyte 0.98 / LOW; review **True** with `HIGH_UNCERTAINTY`, `UNSUPPORTED_CITATION` and `INSUFFICIENT_CELL_COUNT`; `citation_check` = cited [1, 2], retrieved none |

The thesis UI screenshots (Figures 4 and 5) are rendered from the final run in this folder by
`scripts/render_ui_screenshots.py` (the real frontend, fed this saved output).
