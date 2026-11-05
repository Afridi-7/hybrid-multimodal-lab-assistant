# Case-study re-run (thesis Section 6.6)

Output of the final system (v2 classifier, minimum-cell rule, BGE-small embedder) on the
case-study image (bundled sample `BCCD_00010.jpg`, 640x480), run on 5 November 2026 from the
web interface. Copied here because `pipeline.save_intermediate_results: true` makes every run
overwrite `results/stage*.json`.

- An earlier re-run with the corrected classifier (29 September 2026) is documented only by
  `logs/pipeline_20260929_150829.log`; its output files were overwritten by this run. Its
  Stage 1/2 values were identical (2 WBC, 17 RBC; ig 0.42 HIGH, monocyte 0.98 LOW).
- The original run (28 April 2026, v1 classifier, before the minimum-cell rule) is documented by
  `logs/pipeline_20260428_140721.log` and thesis Appendix C.
