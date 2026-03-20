from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from align.io import read_jsonl, write_jsonl


def _load_prompts(path: Path) -> list[dict[str, str]]:
    rows = read_jsonl(path)
    prompts: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        prompt = str(row.get("prompt") or row.get("instruction") or row.get("query") or "").strip()
        if not prompt:
            continue
        prompt_id = str(row.get("id") or f"prompt_{idx:03d}")
        prompts.append({"id": prompt_id, "prompt": prompt})
    if not prompts:
        raise ValueError(f"No valid prompts found in {path}")
    return prompts


def _resolve_generation_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "repetition_penalty": float(args.repetition_penalty),
        "seed": int(args.seed),
    }


def _load_model(model_ref: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_ref)
    adapter_cfg_path = model_path / "adapter_config.json"

    if adapter_cfg_path.exists():
        from peft import PeftModel

        adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
        base_model_ref = adapter_cfg.get("base_model_name_or_path")
        if not base_model_ref:
            raise ValueError(f"Missing base_model_name_or_path in {adapter_cfg_path}")

        tokenizer = AutoTokenizer.from_pretrained(base_model_ref, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_ref,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(model_path))
        resolved_model_ref = f"{base_model_ref} + adapter:{model_ref}"
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        resolved_model_ref = model_ref

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    return tokenizer, model, resolved_model_ref


def _model_device(model) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def _generate_text(tokenizer, model, prompt: str, generation_settings: dict[str, Any]) -> str:
    import torch

    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    input_ids = input_ids.to(_model_device(model))
    attention_mask = torch.ones_like(input_ids)

    kwargs: dict[str, Any] = {
        "max_new_tokens": generation_settings["max_new_tokens"],
        "do_sample": generation_settings["do_sample"],
        "repetition_penalty": generation_settings["repetition_penalty"],
        "pad_token_id": tokenizer.pad_token_id,
    }
    if generation_settings["do_sample"]:
        kwargs["temperature"] = generation_settings["temperature"]
        kwargs["top_p"] = generation_settings["top_p"]

    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    generated = output_ids[0][input_ids.shape[-1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()


def _run_generation(prompts: list[dict[str, str]], model_ref: str, generation_settings: dict[str, Any]) -> list[dict[str, str]]:
    random.seed(generation_settings["seed"])
    try:
        import torch

        torch.manual_seed(generation_settings["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(generation_settings["seed"])
    except ModuleNotFoundError:
        pass

    tokenizer, model, resolved_model_ref = _load_model(model_ref)
    rows: list[dict[str, str]] = []
    for item in prompts:
        rows.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "output": _generate_text(tokenizer, model, item["prompt"], generation_settings),
                "model": resolved_model_ref,
            }
        )
    return rows


def _before_mode(
    prompts_path: Path,
    model_ref: str,
    output_path: Path,
    generation_settings: dict[str, Any],
) -> dict[str, Any]:
    prompts = _load_prompts(prompts_path)
    generated = _run_generation(prompts, model_ref, generation_settings)
    rows = [
        {
            "id": row["id"],
            "prompt": row["prompt"],
            "before_output": row["output"],
            "before_model": row["model"],
            "generation_settings": generation_settings,
        }
        for row in generated
    ]
    write_jsonl(output_path, rows)
    return {"mode": "before", "prompts": len(rows), "output_path": str(output_path)}


def _after_mode(
    prompts_path: Path,
    model_ref: str,
    before_cache: Path,
    output_path: Path,
    generation_settings: dict[str, Any],
) -> dict[str, Any]:
    prompts = _load_prompts(prompts_path)
    before_rows = read_jsonl(before_cache)
    before_by_id = {str(row.get("id", "")): row for row in before_rows}
    generated = _run_generation(prompts, model_ref, generation_settings)

    rows: list[dict[str, Any]] = []
    for row in generated:
        cached = before_by_id.get(row["id"], {})
        rows.append(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "before_output": cached.get("before_output", ""),
                "after_output": row["output"],
                "before_model": cached.get("before_model", ""),
                "after_model": row["model"],
                "generation_settings": generation_settings,
            }
        )
    write_jsonl(output_path, rows)
    return {"mode": "after", "prompts": len(rows), "output_path": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic before/after SFT comparisons.")
    parser.add_argument("--mode", choices=["before", "after"], required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--model", type=str, required=True, help="Model path or model id.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before-cache", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generation_settings = _resolve_generation_settings(args)
    if args.mode == "before":
        summary = _before_mode(
            prompts_path=args.prompts,
            model_ref=args.model,
            output_path=args.output,
            generation_settings=generation_settings,
        )
    else:
        if args.before_cache is None:
            raise ValueError("--before-cache is required for --mode after")
        summary = _after_mode(
            prompts_path=args.prompts,
            model_ref=args.model,
            before_cache=args.before_cache,
            output_path=args.output,
            generation_settings=generation_settings,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
