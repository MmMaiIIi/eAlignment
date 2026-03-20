# JDDC Filtering Relaxation Note

## What Was Too Strict Before

The JDDC path in `align/data.py` rejected rows when category inference failed.  
On local JDDC data, this dominated failures (`10609/10612` rejected rows in one sampled run), even when query/response extraction succeeded.

## What Was Relaxed

1. Category handling:
   - `jddc` rows no longer fail only because category inference is uncertain.
   - If explicit/inferred category is missing, fallback to `general`.
2. Chinese-aware inference:
   - Added Chinese customer-service keywords for refund, shipping, specs, order changes, after-sales, and complaint handling.
3. Dialogue tolerance:
   - Accept broader dialog container keys (`turns`, `utterances`, `chat` in addition to prior keys).
   - Skip malformed turns and keep recoverable user/assistant pairs.
   - If no adjacent pair is found, attempt a fallback pair (last user + next assistant).
4. Text quality guard:
   - Keep short valid replies (`好的`, `稍等`, `可以的`) as valid.
   - Treat empty, pure punctuation, and placeholder-only text as unusable.

## What Is Still Hard-Rejected

- No usable final user query
- No usable assistant response
- Fewer than 2 usable turns in dialogue rows
- Dialogue rows with unparseable turns only
- Any row that cannot be converted into a final SFT record

## Why `general` Fallback Is Safer

For Chinese JDDC, taxonomy misses are often keyword-coverage issues rather than data-quality issues.  
Using `general` keeps high-quality conversational supervision while preserving hard guards for obviously broken samples.

## How To Validate Acceptance Rate

Run preprocessing on local converted input:

```powershell
python scripts/prepare_data.py `
  --profile smoke `
  --input data/raw/jddc.jsonl `
  --source-format jddc `
  --source-name jddc_local `
  --output-dir data/processed `
  --rejected-path data/interim/sft_rejected.jsonl `
  --dataset-info data/processed/dataset_info.json
```

Inspect headline counts from command output:

- `total_raw`
- `valid`
- `rejected`
- `rejection_summary`

Quick check:

- `valid / total_raw` should be substantially higher than the previous low baseline.
- `rejection_summary` should now be dominated by true hard failures (missing/malformed query-response), not category misses.
