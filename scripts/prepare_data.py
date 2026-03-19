from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from align.config import load_profile, resolve
from align.data import SOURCE_FORMATS, prepare_sft_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT data for LLaMA-Factory.")
    parser.add_argument("--profile", type=str, default=None, help="Profile from configs/profiles.yaml")
    parser.add_argument("--input", type=Path, default=None, help="Override raw SFT input JSONL")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override processed output directory")
    parser.add_argument("--rejected-path", type=Path, default=None, help="Override rejected SFT JSONL path")
    parser.add_argument("--dataset-info", type=Path, default=None, help="Override dataset_info.json path")
    parser.add_argument("--source-name", type=str, default=None)
    parser.add_argument("--source-format", type=str, choices=list(SOURCE_FORMATS), default="internal")
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()

    profile_name, profile = load_profile(args.profile)
    split_cfg = profile["split"]
    input_path = args.input or resolve(profile["sft_input"])
    output_dir = args.output_dir or resolve(profile.get("processed_dir", "data/processed"))
    rejected_path = args.rejected_path or resolve(profile.get("sft_rejected", "data/interim/sft_rejected.jsonl"))
    dataset_info = args.dataset_info or resolve(profile.get("dataset_info", "data/processed/dataset_info.json"))
    source_name = args.source_name or f"{profile_name}_sft_source"

    summary = prepare_sft_dataset(
        input_path=input_path,
        output_dir=output_dir,
        rejected_path=rejected_path,
        dataset_info_path=dataset_info,
        split_cfg=split_cfg,
        source_name=source_name,
        source_format=args.source_format,
        fail_on_invalid=args.fail_on_invalid,
    )
    print(json.dumps({"profile": profile_name, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
