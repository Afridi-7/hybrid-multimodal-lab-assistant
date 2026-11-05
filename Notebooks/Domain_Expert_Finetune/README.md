# Domain-Expert Fine-Tuning

Recipe for the open, self-hostable form of the domain expert: **Llama-3.1-8B-Instruct + QLoRA**,
trained on synthetic, textbook-grounded hematology Q&A. It plugs into Stage 3 through the
OpenAI-compatible interface, with no Python changes. Results and discussion: thesis Sections 3.2,
5.3–5.4 and 6.3–6.5.

> **Correction (November 2026).** The first run (`domain_expert_evaluation.ipynb`, results in
> `results/domain_expert/`) used a train/test split that was effectively random, because the
> dataset did not record which textbook chunk each pair came from (see
> `scripts/check_split_leakage.py`). The data were regenerated with chunk labels and the model was
> retrained on a block-level split (`domain_expert_evaluation_v2.ipynb`, results in
> `results/domain_expert_v2/`). Use the v2 workflow below.

## Files

| File | Role |
|---|---|
| [scripts/build_finetune_dataset.py](../../scripts/build_finetune_dataset.py) | Generates 2 Q&A pairs per textbook chunk with an OpenAI model; every record stores its `source`, `chunk_id` and `chunk_key`. Output `data/finetune/train_v2.jsonl`. |
| [domain_expert_evaluation_v2.ipynb](domain_expert_evaluation_v2.ipynb) | Colab (T4): block-level 90/10 split, QLoRA training, evaluation of base vs. fine-tuned (and GPT-4o). |
| [publish_adapter_to_hf.ipynb](publish_adapter_to_hf.ipynb) | Colab (CPU): uploads the v2 adapter and model card to Hugging Face. |
| [scripts/add_gpt4o_baseline.py](../../scripts/add_gpt4o_baseline.py) | Adds GPT-4o answers to a saved report if the Colab step failed. |
| [scripts/judge_domain_expert.py](../../scripts/judge_domain_expert.py) | LLM-as-judge ratings (GPT-4o-mini) of the saved answers. |
| [scripts/analyze_domain_expert.py](../../scripts/analyze_domain_expert.py), [scripts/analyze_judge.py](../../scripts/analyze_judge.py) | All statistics reported in the thesis (ROUGE-L, similarity, CIs, McNemar, citation checks, judge means/SDs/CIs). |
| [scripts/check_v2_split.py](../../scripts/check_v2_split.py) | Rebuilds the v2 split and checks it (no shared chunks, overlap with neighbours, duplicates). |
| [scripts/plot_domain_expert_figures.py](../../scripts/plot_domain_expert_figures.py) | Draws the thesis figures from the saved results. |
| `domain_expert_evaluation.ipynb` | First (leaky) run, kept as a record. |
| `domain_expert_finetune.ipynb` | Early prototype notebook, not used for the thesis results. |

## Step 1 — Build the dataset (local)

Put your OpenAI key in the repo's `.env` file (`OPENAI_API_KEY=...`). Never paste it into a
terminal command or a notebook cell.

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/build_finetune_dataset.py            # ~2 USD with gpt-4o-mini
python scripts/check_v2_split.py                    # optional: inspect the split it will produce
```

The run should end with `chunks with fewer than 2 pairs: 0` and `Writing 2098 records`.

## Step 2 — Train and evaluate (Colab)

1. Upload `domain_expert_evaluation_v2.ipynb` to Colab; Runtime → Change runtime type → **T4 GPU**.
2. In the 🔑 Secrets panel, add `OPENAI_API_KEY` (a current key) and switch on notebook access.
3. Runtime → Run all; upload `data/finetune/train_v2.jsonl` when asked (~2 h in total).
4. Copy the files of the Drive folder `thesis_domain_expert_v2` (not the `hema-llama-lora` folder)
   into `results/domain_expert_v2/`.

The split is made over **blocks of 10 consecutive chunks** of the same book, because neighbouring
chunks share 40 words. Do not split by position in the file.

## Step 3 — Judge and analyse (local)

```powershell
python scripts/add_gpt4o_baseline.py      # only if GPT-4o is missing from the report
python scripts/judge_domain_expert.py --report results/domain_expert_v2/evaluation_report.json --out results/domain_expert_v2/judge_scores.json
python scripts/analyze_domain_expert.py
python scripts/analyze_judge.py
python scripts/plot_domain_expert_figures.py
```

Check `results/domain_expert_v2/flagged_citations.csv` by hand before reporting the number of
invented references.

## Step 4 — Publish the adapter

Run `publish_adapter_to_hf.ipynb` in Colab (CPU runtime). It reads the adapter from
`thesis_domain_expert_v2/hema-llama-lora` in Drive and asks for a Hugging Face **write** token in a
hidden box.

## Step 5 — Serve and plug in

On any machine with a GPU (≥10 GB for 8B in 4-bit):
```bash
pip install vllm
vllm serve Afridi07/hematology-llama-3.1-8b-lora --port 8001 --enable-lora
```

Then in `config.yaml`:
```yaml
llm:
  model_name: Afridi07/hematology-llama-3.1-8b-lora
  temperature: 0.1
```
and in `.env`:
```
OPENAI_API_KEY=dummy
OPENAI_BASE_URL=http://localhost:8001/v1
```

Note: the fine-tuned model was evaluated stand-alone in the thesis, not inside the agent.
