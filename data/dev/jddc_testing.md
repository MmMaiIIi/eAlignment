# JDDC Integration Testing (Local Repo)

## Purpose in This Repo

In `eAlignment`, JDDC is an **external SFT source** consumed by:

- `scripts/prepare_data.py` (entrypoint)
- `align/data.py` (`source_format=jddc` normalizer)

This repo only normalizes/validates/splits data. Training stays in LLaMA-Factory.

## Local Dataset Location and Raw Source

- Downloaded dataset folder:
  - `data/JDDC-Baseline-Seq2Seq-master/`
- Local raw source file used for integration:
  - `data/JDDC-Baseline-Seq2Seq-master/data/chat.txt`

`chat.txt` is tab-separated session-turn text (header includes `session_id`, `waiter_send`, and content columns), not JSONL.

## Is `chat.txt` Directly Consumable?

No. `scripts/prepare_data.py` reads input with JSONL parsing (`json.loads` per line).  
`chat.txt` must be converted into JSONL rows before `--source-format jddc` can normalize them.

Recommended staging flow:

1. Source dataset:
   - `data/JDDC-Baseline-Seq2Seq-master/data/chat.txt`
2. Optional conversion output:
   - `data/raw/jddc.jsonl`
3. Normalized/validated outputs:
   - `data/interim/*`
   - `data/processed/*`

## Current JDDC Normalizer Input Contract

Each JSONL row should be a JSON object. For dialogs, these keys are accepted:

- Dialog container keys: `dialog`, `dialogue`, `conversation`, `messages`, `session`
- Turn role keys: `role`, `speaker`, `from`, `type`
- Turn text keys: `text`, `content`, `utterance`, `sentence`, `msg`, `message`

Supported turn role aliases:

- User side: `user`, `customer`, `buyer`, `q`, `human`
- Assistant side: `assistant`, `agent`, `seller`, `a`, `bot`

Also supported turn shapes:

- Mapping turn: `{"role": "...", "text": "..."}`
- Compact list turn: `["q", "..."]` / `["a", "..."]`
- String turn prefixes:
  - user: `q:`, `user:`, `buyer:`, `customer:`
  - assistant: `a:`, `assistant:`, `seller:`, `agent:`

Fallback direct fields if dialog extraction fails:

- Query keys: `query`, `question`, `customer_query`, `instruction`
- Response keys: `response`, `answer`, `assistant_response`, `reply`
- Context keys: `context`, `history`, `input`
- ID keys: `id`, `session_id`, `sessionid`, `dialog_id`, `dialogue_id`

## How Query/Context/Response Are Extracted

For dialog-style rows, the normalizer:

1. Keeps only turns with recognized role + non-empty text.
2. Finds the **last adjacent** `user -> assistant` pair.
3. Uses that pair as:
   - final user query
   - assistant response
4. Builds prior context from turns before that pair, as newline-joined:
   - `user: ...`
   - `assistant: ...`

During final SFT parsing, whitespace is normalized, so multiline context is persisted as a single-space-separated string in processed outputs.

If no usable pair is found and no fallback fields are provided, row is rejected with normalize errors (for example missing query/response text).

## Required Fields After Normalization

Normalizer output (internal normalized raw):

- `id`, `category`, `system`, `query`, `input`, `response`, `source`, `source_id`

Validated SFT output in `data/processed/sft_*.jsonl`:

- `id`, `category`, `system`, `instruction`, `input`, `output`, `source`, `source_id`

## Category Behavior and Limitations

Explicit category aliases are normalized from keys like `category`/`intent`/`topic`/`domain`:

- returns/refunds aliases -> `returns_refunds`
- shipping/logistics aliases -> `shipping_logistics`
- specs/product aliases -> `product_specs`
- order change aliases -> `order_modification`
- warranty/aftersales aliases -> `after_sales`
- complaint/deescalation aliases -> `complaint_soothing`

If explicit category is absent/unsupported, text-based inference uses keyword rules:

- refund/return/rma
- shipping/delivery/logistics/tracking
- spec/size/material/compatib
- change/modify/cancel/address
- warranty/repair/after-sales/aftersales
- complaint/angry/frustrated/escalat

Limitation: inference is keyword-based and mostly English-oriented; raw Chinese-only rows often fail inference unless conversion injects category.

## Minimal Conversion Step (Current Gap Bridging)

If no dedicated converter script is used, this PowerShell inline Python converts `chat.txt` -> `data/raw/jddc.jsonl` by session:

```powershell
@'
import csv, json
from collections import defaultdict
from pathlib import Path

src = Path(r"data/JDDC-Baseline-Seq2Seq-master/data/chat.txt")
dst = Path(r"data/raw/jddc.jsonl")

sessions = defaultdict(list)
with src.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        sid = (row.get("session_id") or "").strip()
        text = (row.get("") or row.get("content") or "").strip()
        if not sid or not text:
            continue
        role = "assistant" if (row.get("waiter_send", "").strip() == "1") else "user"
        sessions[sid].append({"role": role, "text": text})

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8", newline="\n") as f:
    for sid, dialog in sessions.items():
        f.write(json.dumps({"session_id": sid, "dialog": dialog}, ensure_ascii=False) + "\n")

print(f"wrote {len(sessions)} sessions -> {dst}")
'@ | python -
```

Note: this is a minimal bridge; category assignment is still needed for high acceptance.

## Smoke Commands (Current Local Layout)

Prepare from converted JSONL:

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

Optional strict mode (fail if any invalid row):

```powershell
python scripts/prepare_data.py `
  --profile smoke `
  --input data/raw/jddc.jsonl `
  --source-format jddc `
  --source-name jddc_local `
  --fail-on-invalid
```

## Artifact Placement

- `data/raw/`:
  - source JSONL for this repo (for example `jddc.jsonl`)
- `data/interim/`:
  - rejected records (`sft_rejected.jsonl`) with `dataset`, `line_no`, `source`, `errors`, and original `raw`
- `data/processed/`:
  - accepted SFT files (`sft_all/train/dev/test.jsonl`)
  - `dataset_info.json`

## Rejected-Data Files

Rejected SFT rows are written to `data/interim/sft_rejected.jsonl`.  
Common JDDC normalize errors:

- `normalize: jddc: missing query text`
- `normalize: jddc: missing response text`
- `normalize: jddc: missing/unsupported category and inference failed`

## Failure Modes and Debugging

1. `chat.txt` passed directly to `--input`:
   - symptom: JSON decode failure before rejection file writing
   - fix: convert to JSONL first
2. All rows rejected for category:
   - symptom: `missing/unsupported category and inference failed`
   - fix: add explicit category during conversion, or improve upstream mapping
3. Missing query/response after conversion:
   - symptom: missing query/response normalize errors
   - fix: ensure dialogs contain at least one usable `user -> assistant` pair
4. Unexpected role labels:
   - symptom: turns ignored, then missing pair/query
   - fix: map roles to supported aliases listed above

## Acceptance Checklist Before Large-Scale Run

- `chat.txt` converted into JSONL rows under `data/raw/` (for example `data/raw/jddc.jsonl`)
- Rows follow current `jddc` contract (`dialog`/turn shape and role aliases)
- Category strategy is decided (explicit mapping preferred for Chinese dialogs)
- Smoke run completes and writes both processed + rejected artifacts
- Rejected ratio and top error reasons reviewed in `data/interim/sft_rejected.jsonl`
- `data/processed/dataset_info.json` exists and matches expected Alpaca mapping
