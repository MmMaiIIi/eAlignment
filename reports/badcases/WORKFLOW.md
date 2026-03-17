# Badcase Workflow

## Purpose

Turn evaluation failures into actionable model/data improvements without building a heavy platform.

## Inputs

- `reports/experiments/latest_eval/per_sample.jsonl`
- `reports/experiments/latest_eval/badcases.jsonl`
- `reports/experiments/latest_eval/summary.json`

## Steps

1. Run evaluation and badcase extraction.
2. Run `scripts/summarize_badcases.py` to produce a quick shortlist.
3. Copy each shortlisted case into `reports/templates/badcase_analysis_template.md`.
4. Tag root causes:
   - data_gap
   - policy_violation
   - tone_issue
   - actionability_gap
   - hallucination
   - prompt_mismatch
5. Write one proposed fix per case:
   - data augmentation
   - stronger rejection sampling
   - preference cleanup
   - config adjustment
6. Link fixes to planned ablations in `reports/experiments/ablation_plan.md`.

## Output Artifacts

- `reports/badcases/latest_badcases.md`
- per-case markdown notes under `reports/badcases/`
- experiment follow-up notes under `reports/experiments/`

## Notes

- Do not claim improvement until rerun metrics confirm it.
- Keep proxy metrics and qualitative judgments clearly separated.
