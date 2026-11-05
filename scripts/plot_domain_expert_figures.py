"""Draw the two domain-expert figures of the thesis from saved result files.

  thesis/figures/domain_expert_results.pdf   <- analysis.json + judge_analysis.json
  thesis/figures/qlora_training_loss.pdf     <- train_log.json

Usage:  python scripts/plot_domain_expert_figures.py [--run results/domain_expert_v2]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE, OURS, GPT = "Llama-3.1-8B (base)", "Llama-3.1-8B + QLoRA (ours)", "GPT-4o (reference)"
LABEL = {BASE: "Llama-3.1-8B\n(base)", OURS: "Llama-3.1-8B\n+ QLoRA (ours)", GPT: "GPT-4o\n(reference)"}
COLOR = {BASE: "#a8a59c", OURS: "#2c7be5", GPT: "#ec6632"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def bars(ax, systems, values, fmt, title, ylim):
    xs = range(len(systems))
    ax.bar(xs, values, width=0.6, color=[COLOR[s] for s in systems])
    for x, v in zip(xs, values):
        ax.text(x, v + ylim[1] * 0.02, fmt(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(xs)); ax.set_xticklabels([LABEL[s] for s in systems], fontsize=7.5)
    ax.set_ylim(*ylim); ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("results/domain_expert_v2"))
    ap.add_argument("--out", type=Path, default=Path("thesis/figures"))
    a = ap.parse_args()
    an = json.loads((a.run / "analysis.json").read_text(encoding="utf-8"))
    ju = json.loads((a.run / "judge_analysis.json").read_text(encoding="utf-8"))
    log = json.loads((a.run / "train_log.json").read_text(encoding="utf-8"))
    sysn = [s for s in (BASE, OURS, GPT) if s in an["systems"]]

    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.4))
    bars(axs[0, 0], sysn, [an["systems"][s]["semantic_sim"] for s in sysn], lambda v: f"{v:.3f}",
         "(a) Semantic similarity to reference", (0, 1))
    bars(axs[0, 1], sysn, [an["systems"][s]["rouge_l"] for s in sysn], lambda v: f"{v:.3f}",
         "(b) ROUGE-L F1 vs. reference", (0, 1))
    ax = axs[1, 0]; w = 0.25
    for k, s in enumerate(sysn):
        vals = [ju["systems"][s][m]["mean"] for m in ("faithfulness", "clinical_correctness")]
        xs = [i + (k - 1) * w for i in range(2)]
        ax.bar(xs, [v - 1 for v in vals], bottom=1, width=w, color=COLOR[s], label=s)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["faithfulness", "clinical correctness"])
    ax.set_ylim(1, 5.3); ax.legend(fontsize=6.5, loc="upper left", frameon=False)
    ax.set_title(f"(c) LLM-judge scores (1–5, {ju.get('judge_model', 'judge')})", fontsize=9)
    mq = an["medmcqa"]
    bars(axs[1, 1], [BASE, OURS], [mq["acc_base_pct"], mq["acc_ft_pct"]], lambda v: f"{v:.0f}%",
         "(d) MedMCQA accuracy, %", (0, 100))
    axs[1, 1].axhline(25, ls="--", lw=0.8, color="grey"); axs[1, 1].text(0.5, 26.5, "chance (25%)", fontsize=6.5, color="grey", ha="center")
    fig.tight_layout(); a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / "domain_expert_results.pdf"); plt.close(fig)

    steps = [h["step"] for h in log]; loss = [h["loss"] for h in log]
    end1 = log[0]["step"] / log[0]["epoch"]          # optimisation steps per epoch (logged ratio)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.plot(steps, loss, color=COLOR[OURS], lw=1.4)
    ax.axvline(end1, ls="--", lw=0.8, color="grey"); ax.text(end1 + 4, max(loss) * 0.9, "end of epoch 1", fontsize=7.5, color="grey")
    ax.set_xlabel("optimisation step (batch of 8 examples)"); ax.set_ylabel("training loss")
    fig.tight_layout(); fig.savefig(a.out / "qlora_training_loss.pdf"); plt.close(fig)
    print("written:", a.out / "domain_expert_results.pdf", a.out / "qlora_training_loss.pdf")


if __name__ == "__main__":
    main()
