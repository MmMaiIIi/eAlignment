from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import re
import unicodedata
from typing import Any, Mapping

from align.io import read_jsonl, write_json, write_jsonl
from align.prompts import SYSTEM_PROMPT


CATEGORIES = [
    "returns_refunds",
    "shipping_logistics",
    "product_specs",
    "order_modification",
    "after_sales",
    "complaint_soothing",
    "general",
]

SOURCE_FORMATS = ("internal", "jddc", "ecd", "faq")

CONTEXT_WINDOW_TURNS = 8

PLACEHOLDER_TOKENS = {
    "null",
    "none",
    "nan",
    "n/a",
    "na",
    "unknown",
    "unknown_text",
    "无",
    "暂无",
    "空",
    "未知",
    "待补充",
}

JDDC_DIALOG_KEYS = ["dialog", "dialogue", "conversation", "messages", "session", "turns", "utterances", "chat"]

CATEGORY_ALIASES = {
    "returns_refunds": "returns_refunds",
    "returns": "returns_refunds",
    "refund": "returns_refunds",
    "refunds": "returns_refunds",
    "return": "returns_refunds",
    "return_refund": "returns_refunds",
    "shipping_logistics": "shipping_logistics",
    "shipping": "shipping_logistics",
    "logistics": "shipping_logistics",
    "delivery": "shipping_logistics",
    "product_specs": "product_specs",
    "product_spec": "product_specs",
    "specs": "product_specs",
    "product": "product_specs",
    "product_consultation": "product_specs",
    "consultation": "product_specs",
    "order_modification": "order_modification",
    "order_change": "order_modification",
    "change_order": "order_modification",
    "modify_order": "order_modification",
    "after_sales": "after_sales",
    "aftersales": "after_sales",
    "post_sale": "after_sales",
    "warranty": "after_sales",
    "warranty_repair": "after_sales",
    "complaint_soothing": "complaint_soothing",
    "complaint": "complaint_soothing",
    "deescalation": "complaint_soothing",
    "complaint_resolution": "complaint_soothing",
    "general": "general",
    "other": "general",
    "others": "general",
    "misc": "general",
    "miscellaneous": "general",
    "退款": "returns_refunds",
    "退货": "returns_refunds",
    "退換": "returns_refunds",
    "物流": "shipping_logistics",
    "配送": "shipping_logistics",
    "发货": "shipping_logistics",
    "規格": "product_specs",
    "规格": "product_specs",
    "咨询": "product_specs",
    "改地址": "order_modification",
    "修改订单": "order_modification",
    "售后": "after_sales",
    "维修": "after_sales",
    "投诉": "complaint_soothing",
    "其他": "general",
}

SFT_KEYS = {
    "id": ["id", "sample_id", "record_id", "uid"],
    "category": ["category", "intent", "topic"],
    "system": ["system", "system_prompt", "sys_prompt"],
    "instruction": ["instruction", "query", "customer_query", "prompt", "user_message"],
    "input": ["input", "context", "extra_context"],
    "output": ["output", "response", "assistant_response", "answer"],
    "source": ["source", "source_name"],
    "source_id": ["source_id", "original_id", "ticket_id"],
}

PREF_KEYS = {
    "id": ["id", "sample_id", "record_id", "uid"],
    "category": ["category", "intent", "topic"],
    "prompt": ["prompt", "instruction", "query", "user_message"],
    "chosen": ["chosen", "preferred", "good_response"],
    "rejected": ["rejected", "dispreferred", "bad_response"],
    "source": ["source", "source_name"],
    "source_id": ["source_id", "original_id", "ticket_id"],
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _string_text(value: Any) -> str:
    if isinstance(value, str):
        return _text(value)
    return ""


def _strip_jddc_placeholders(value: str) -> str:
    text = _text(value)
    text = re.sub(r"#?e-s\[[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[a-z_]+\w*\]", "", text, flags=re.IGNORECASE)
    return _text(text)


def _has_substantive_char(value: str) -> bool:
    for ch in value:
        if ch.isspace():
            continue
        category = unicodedata.category(ch)
        if category and category[0] not in {"P", "S"}:
            return True
    return False


def _is_usable_text(value: Any) -> bool:
    text = _text(value)
    if not text:
        return False
    token = text.lower()
    if token in PLACEHOLDER_TOKENS:
        return False
    stripped = _strip_jddc_placeholders(text)
    if not stripped:
        return False
    if stripped.lower() in PLACEHOLDER_TOKENS:
        return False
    return _has_substantive_char(stripped)


def _category(value: Any) -> str:
    token = _text(value).lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    return CATEGORY_ALIASES.get(token, "")


def _pick(raw: Mapping[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    return default


def _pick_text(raw: Mapping[str, Any], keys: list[str], default: str = "") -> str:
    for key in keys:
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, str):
            return _text(value)
    return default


def _require_text(record: Mapping[str, Any], fields: list[str], allow_empty: set[str] | None = None) -> list[str]:
    allow_empty = allow_empty or set()
    errors: list[str] = []
    for field in fields:
        value = record.get(field)
        if not isinstance(value, str):
            errors.append(f"{field}: must be string")
            continue
        if field not in allow_empty and not value.strip():
            errors.append(f"{field}: must be non-empty")
    return errors


def _split(records: list[dict[str, Any]], train: float, dev: float, test: float, seed: int) -> dict[str, list[dict[str, Any]]]:
    if abs((train + dev + test) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")
    rows = list(records)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_train = int(n * train)
    n_dev = int(n * dev)
    if n > 0 and n_train == 0:
        n_train = 1
    if n_train + n_dev > n:
        n_dev = max(0, n - n_train)
    n_test = n - n_train - n_dev
    if n >= 3 and n_dev == 0:
        n_dev = 1
        n_train = max(n_train - 1, 1)
        n_test = n - n_train - n_dev
    if n >= 3 and n_test == 0:
        n_test = 1
        n_train = max(n_train - 1, 1)
    return {
        "train": rows[:n_train],
        "dev": rows[n_train : n_train + n_dev],
        "test": rows[n_train + n_dev : n_train + n_dev + n_test],
    }


def _rejected(raw: Mapping[str, Any], line_no: int, errors: list[str], source_name: str, dataset: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "line_no": line_no,
        "source": source_name,
        "errors": errors,
        "raw": dict(raw),
    }


def _infer_category_from_text(value: str) -> str:
    text = value.lower()
    rules = [
        (["refund", "return", "rma", "退款", "退货", "换货", "补偿", "价保"], "returns_refunds"),
        (["shipping", "delivery", "logistics", "tracking", "物流", "发货", "快递", "配送", "运单", "签收", "催单"], "shipping_logistics"),
        (["spec", "size", "material", "compatib", "规格", "尺寸", "材质", "型号", "参数", "兼容", "咨询"], "product_specs"),
        (["change", "modify", "cancel", "address", "修改", "取消", "改地址", "改订单", "收货地址", "订单信息"], "order_modification"),
        (["warranty", "repair", "after-sales", "aftersales", "保修", "维修", "售后", "故障", "坏了", "换新"], "after_sales"),
        (["complaint", "angry", "frustrated", "escalat", "投诉", "生气", "不满", "人工", "升级处理", "差评"], "complaint_soothing"),
    ]
    for keywords, category in rules:
        if any(keyword in text for keyword in keywords):
            return category
    return ""


def _turn_role(turn: Any) -> str:
    if isinstance(turn, Mapping):
        role = _text(_pick(turn, ["role", "speaker", "from", "type"])).lower()
        if role in {"customer", "user", "buyer", "q", "human"}:
            return "user"
        if role in {"assistant", "agent", "seller", "a", "bot"}:
            return "assistant"
        waiter_send = _text(_pick(turn, ["waiter_send", "is_waiter", "agent_send", "seller_send"])).lower()
        if waiter_send in {"1", "true", "yes"}:
            return "assistant"
        if waiter_send in {"0", "false", "no"}:
            return "user"
        customer_send = _text(_pick(turn, ["is_customer", "from_customer", "customer_send"])).lower()
        if customer_send in {"1", "true", "yes"}:
            return "user"
        if customer_send in {"0", "false", "no"}:
            return "assistant"
        sender = _text(_pick(turn, ["sender", "speaker_id", "party"])).lower()
        if sender in {"0", "user", "customer", "buyer", "human"}:
            return "user"
        if sender in {"1", "assistant", "agent", "seller", "bot"}:
            return "assistant"
        return ""
    if isinstance(turn, list) and turn:
        role = _text(turn[0]).lower()
        if role in {"customer", "user", "buyer", "q", "human"}:
            return "user"
        if role in {"assistant", "agent", "seller", "a", "bot"}:
            return "assistant"
    if isinstance(turn, str):
        text = turn.strip().lower()
        if text.startswith(("q:", "user:", "buyer:", "customer:", "问:", "用户:", "买家:", "顾客:")):
            return "user"
        if text.startswith(("a:", "assistant:", "seller:", "agent:", "答:", "客服:", "商家:", "店铺:")):
            return "assistant"
    return ""


def _turn_text(turn: Any) -> str:
    if isinstance(turn, Mapping):
        return _string_text(_pick(turn, ["text", "content", "utterance", "sentence", "msg", "message"]))
    if isinstance(turn, list) and len(turn) >= 2:
        return _string_text(turn[1])
    if isinstance(turn, str):
        return _text(turn)
    return ""


def _dialog_triplet(value: Any) -> tuple[str, str, str, dict[str, Any]]:
    meta = {
        "has_dialog": isinstance(value, list),
        "total_turns": 0,
        "usable_turns": 0,
        "used_fallback_pair": False,
    }
    if not isinstance(value, list):
        return "", "", "", meta
    turns: list[tuple[str, str]] = []
    for turn in value:
        meta["total_turns"] += 1
        role = _turn_role(turn)
        text = _turn_text(turn)
        if role and _is_usable_text(text):
            turns.append((role, text))
    meta["usable_turns"] = len(turns)
    if not turns:
        return "", "", "", meta
    if len(turns) < 2:
        return "", "", "", meta

    user_idx = -1
    assistant_idx = -1
    for idx in range(len(turns) - 1):
        if turns[idx][0] == "user" and turns[idx + 1][0] == "assistant":
            user_idx = idx
            assistant_idx = idx + 1
    if user_idx == -1:
        # Adjacent user->assistant pairs were previously required, which was too strict for noisy JDDC rows.
        for idx in range(len(turns) - 1, -1, -1):
            if turns[idx][0] != "user":
                continue
            for nxt in range(idx + 1, len(turns)):
                if turns[nxt][0] == "assistant":
                    user_idx = idx
                    assistant_idx = nxt
                    meta["used_fallback_pair"] = True
                    break
            if user_idx != -1:
                break
    if user_idx == -1 or assistant_idx == -1:
        return "", "", "", meta

    instruction = turns[user_idx][1]
    output = turns[assistant_idx][1]
    start = max(0, user_idx - CONTEXT_WINDOW_TURNS)
    context_parts = [f"{role}: {text}" for role, text in turns[start:user_idx]]
    return instruction, "\n".join(context_parts), output, meta


def _normalize_external_sft(
    raw: Mapping[str, Any], source_name: str, source_format: str, line_no: int
) -> dict[str, Any]:
    if source_format not in SOURCE_FORMATS:
        raise ValueError(f"Unsupported source_format `{source_format}`. Expected one of {SOURCE_FORMATS}.")
    if source_format == "internal":
        return dict(raw)

    if source_format == "jddc":
        dialog_value = _pick(raw, JDDC_DIALOG_KEYS)
        query, context, response, dialog_meta = _dialog_triplet(dialog_value)
        if not _is_usable_text(query):
            query = _pick_text(raw, ["query", "question", "customer_query", "instruction"])
        if not _is_usable_text(response):
            response = _pick_text(raw, ["response", "answer", "assistant_response", "reply"])
        input_text = _pick_text(raw, ["context", "history", "input"]) or context
        record_id = _text(_pick(raw, ["id", "session_id", "sessionid", "dialog_id", "dialogue_id"]))
    elif source_format == "ecd":
        dialog_meta = {"has_dialog": False, "total_turns": 0, "usable_turns": 0, "used_fallback_pair": False}
        query = _pick_text(raw, ["buyer_query", "customer_query", "user_query", "question", "instruction", "query"])
        response = _pick_text(raw, ["seller_response", "agent_response", "assistant_response", "answer", "response"])
        input_text = _pick_text(raw, ["context", "history", "input"])
        if not query or not response:
            dialog_query, dialog_context, dialog_response, dialog_meta = _dialog_triplet(
                _pick(raw, ["dialog", "dialogue", "conversation", "messages", "turns", "utterances"])
            )
            query = query or dialog_query
            response = response or dialog_response
            input_text = input_text or dialog_context
        record_id = _text(_pick(raw, ["id", "sample_id", "record_id", "uid", "session_id"]))
    else:  # faq
        dialog_meta = {"has_dialog": False, "total_turns": 0, "usable_turns": 0, "used_fallback_pair": False}
        query = _pick_text(raw, ["question", "faq_question", "query", "instruction", "title"])
        response = _pick_text(raw, ["answer", "faq_answer", "response", "output", "content"])
        input_text = _pick_text(raw, ["context", "detail", "input"])
        record_id = _text(_pick(raw, ["id", "faq_id", "question_id", "uid", "record_id"]))

    if not _is_usable_text(query):
        if source_format == "jddc" and dialog_meta["has_dialog"]:
            if dialog_meta["usable_turns"] == 0 and dialog_meta["total_turns"] > 0:
                raise ValueError("jddc: unparseable turns (0 usable)")
            if dialog_meta["usable_turns"] < 2:
                raise ValueError("jddc: malformed dialogue (<2 usable turns)")
            raise ValueError("jddc: malformed dialogue (no recoverable user->assistant pair)")
        raise ValueError(f"{source_format}: missing query text")
    if not _is_usable_text(response):
        if source_format == "jddc" and dialog_meta["has_dialog"] and dialog_meta["usable_turns"] < 2:
            raise ValueError("jddc: malformed dialogue (<2 usable turns)")
        raise ValueError(f"{source_format}: missing response text")

    query = _text(query)
    response = _text(response)

    category_value = _pick_text(raw, ["category", "intent", "topic", "domain", "scene", "label"])
    category = _category(category_value)
    if not category:
        category = _infer_category_from_text(f"{query}\n{input_text}\n{response}")
    if not category and source_format == "faq":
        category = "product_specs"
    if not category:
        # Keep recoverable rows instead of dropping Chinese JDDC records on taxonomy misses.
        category = "general"

    source_id = _text(_pick(raw, ["source_id", "original_id", "ticket_id", "dialog_id", "session_id"])) or record_id
    return {
        "id": record_id or f"{source_format}_{line_no:08d}",
        "category": category,
        "system": _text(_pick(raw, ["system", "system_prompt", "sys_prompt"], SYSTEM_PROMPT)) or SYSTEM_PROMPT,
        "query": query,
        "input": input_text,
        "response": response,
        "source": _text(_pick(raw, ["source", "source_name"], source_name)) or source_name,
        "source_id": source_id,
    }


def _parse_sft(raw: Mapping[str, Any], source_name: str) -> tuple[dict[str, Any], list[str]]:
    record = {
        "id": _text(_pick(raw, SFT_KEYS["id"])),
        "category": _category(_pick(raw, SFT_KEYS["category"])),
        "system": _text(_pick(raw, SFT_KEYS["system"], SYSTEM_PROMPT)),
        "instruction": _text(_pick(raw, SFT_KEYS["instruction"])),
        "input": _text(_pick(raw, SFT_KEYS["input"], "")),
        "output": _text(_pick(raw, SFT_KEYS["output"])),
        "source": _text(_pick(raw, SFT_KEYS["source"], source_name)) or source_name,
        "source_id": _text(_pick(raw, SFT_KEYS["source_id"], "")),
    }
    errors = _require_text(record, ["id", "category", "system", "instruction", "input", "output"], {"input"})
    if record["category"] not in CATEGORIES:
        errors.append(f"category: unsupported `{record['category']}`")
    return record, errors


def _parse_pref(raw: Mapping[str, Any], source_name: str) -> tuple[dict[str, Any], list[str]]:
    record = {
        "id": _text(_pick(raw, PREF_KEYS["id"])),
        "category": _category(_pick(raw, PREF_KEYS["category"])),
        "prompt": _text(_pick(raw, PREF_KEYS["prompt"])),
        "chosen": _text(_pick(raw, PREF_KEYS["chosen"])),
        "rejected": _text(_pick(raw, PREF_KEYS["rejected"])),
        "source": _text(_pick(raw, PREF_KEYS["source"], source_name)) or source_name,
        "source_id": _text(_pick(raw, PREF_KEYS["source_id"], "")),
    }
    errors = _require_text(record, ["id", "category", "prompt", "chosen", "rejected"])
    if record["category"] not in CATEGORIES:
        errors.append(f"category: unsupported `{record['category']}`")
    if record["chosen"].strip() == record["rejected"].strip():
        errors.append("chosen/rejected: chosen and rejected must differ")
    return record, errors


def _quality_report(raw_count: int, valid_rows: list[dict[str, Any]], rejected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    issue_counts = Counter(
        {"empty_chosen_or_rejected": 0, "chosen_identical_to_rejected": 0, "malformed_examples": 0}
    )
    for rejected in rejected_rows:
        errors = [str(err).lower() for err in rejected.get("errors", [])]
        issue_counts["malformed_examples"] += 1
        if any(("chosen" in err or "rejected" in err) and "non-empty" in err for err in errors):
            issue_counts["empty_chosen_or_rejected"] += 1
        if any("chosen/rejected" in err and "must differ" in err for err in errors):
            issue_counts["chosen_identical_to_rejected"] += 1

    categories = Counter(str(row.get("category", "unknown")) for row in valid_rows)
    max_count = max(categories.values()) if categories else 0
    min_count = min(categories.values()) if categories else 0
    imbalance = (max_count / min_count) if min_count > 0 else 0.0

    prompt_map: dict[str, list[str]] = {}
    pair_map: dict[str, list[str]] = {}
    for idx, row in enumerate(valid_rows):
        row_id = row.get("id") or f"pref_{idx:04d}"
        prompt = str(row.get("prompt", "")).strip().lower()
        pair = f"{prompt}|||{str(row.get('chosen', '')).strip().lower()}|||{str(row.get('rejected', '')).strip().lower()}"
        prompt_map.setdefault(prompt, []).append(str(row_id))
        pair_map.setdefault(pair, []).append(str(row_id))

    dup_prompts = [ids for key, ids in prompt_map.items() if key and len(ids) > 1]
    dup_pairs = [ids for key, ids in pair_map.items() if key and len(ids) > 1]

    return {
        "dataset": "dpo",
        "total_raw": raw_count,
        "valid": len(valid_rows),
        "rejected": len(rejected_rows),
        "valid_ratio": (len(valid_rows) / raw_count) if raw_count else 0.0,
        "issue_counts": dict(issue_counts),
        "category_distribution": dict(categories),
        "category_imbalance": {
            "max_count": max_count,
            "min_count": min_count,
            "imbalance_ratio": imbalance,
            "flagged": imbalance >= 2.5 if min_count > 0 else False,
        },
        "duplicate_patterns": {
            "duplicate_prompt_groups": len(dup_prompts),
            "duplicate_pair_groups": len(dup_pairs),
            "duplicate_prompt_examples": dup_prompts[:5],
            "duplicate_pair_examples": dup_pairs[:5],
        },
    }


def _write_dataset_info(path: Path) -> None:
    data = {
        "ecom_sft_seed": {
            "file_name": "sft_train.jsonl",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        },
        "ecom_pref_seed": {
            "file_name": "dpo_train.jsonl",
            "ranking": True,
            "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
        },
    }
    write_json(path, data)


def _sft_rejection_breakdown(rejected_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        {
            "missing_query": 0,
            "missing_response": 0,
            "malformed_dialogue": 0,
            "unparseable_turns": 0,
            "other_hard_failure": 0,
        }
    )
    for row in rejected_rows:
        errors = [str(err).lower() for err in row.get("errors", [])]
        text = " | ".join(errors)
        if "missing query text" in text:
            counts["missing_query"] += 1
        elif "missing response text" in text:
            counts["missing_response"] += 1
        elif "malformed dialogue" in text:
            counts["malformed_dialogue"] += 1
        elif "unparseable turns" in text:
            counts["unparseable_turns"] += 1
        else:
            counts["other_hard_failure"] += 1
    return dict(counts)


def prepare_sft_dataset(
    input_path: Path,
    output_dir: Path,
    rejected_path: Path,
    dataset_info_path: Path | None,
    split_cfg: Mapping[str, Any],
    source_name: str,
    source_format: str = "internal",
    fail_on_invalid: bool = False,
) -> dict[str, Any]:
    raw_rows = read_jsonl(input_path)
    valid_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(raw_rows, start=1):
        try:
            normalized_raw = _normalize_external_sft(
                raw, source_name=source_name, source_format=source_format, line_no=line_no
            )
        except ValueError as exc:
            rejected_rows.append(_rejected(raw, line_no, [f"normalize: {exc}"], source_name, "sft"))
            continue
        parsed, errors = _parse_sft(normalized_raw, source_name=source_name)
        if errors:
            rejected_rows.append(_rejected(raw, line_no, errors, source_name, "sft"))
        else:
            valid_rows.append(parsed)

    if fail_on_invalid and rejected_rows:
        raise ValueError(f"Found {len(rejected_rows)} invalid SFT rows.")

    splits = _split(
        valid_rows,
        train=float(split_cfg["train"]),
        dev=float(split_cfg["dev"]),
        test=float(split_cfg["test"]),
        seed=int(split_cfg["seed"]),
    )
    write_jsonl(output_dir / "sft_all.jsonl", valid_rows)
    write_jsonl(output_dir / "sft_train.jsonl", splits["train"])
    write_jsonl(output_dir / "sft_dev.jsonl", splits["dev"])
    write_jsonl(output_dir / "sft_test.jsonl", splits["test"])
    write_jsonl(rejected_path, rejected_rows)
    if dataset_info_path is not None:
        _write_dataset_info(dataset_info_path)

    return {
        "dataset": "sft",
        "input": str(input_path),
        "source_format": source_format,
        "total_raw": len(raw_rows),
        "valid": len(valid_rows),
        "rejected": len(rejected_rows),
        "rejection_summary": _sft_rejection_breakdown(rejected_rows),
        "train": len(splits["train"]),
        "dev": len(splits["dev"]),
        "test": len(splits["test"]),
    }


def prepare_preference_dataset(
    input_path: Path,
    output_dir: Path,
    rejected_path: Path,
    quality_path: Path,
    dataset_info_path: Path,
    split_cfg: Mapping[str, Any],
    source_name: str,
    fail_on_invalid: bool = False,
) -> dict[str, Any]:
    raw_rows = read_jsonl(input_path)
    valid_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(raw_rows, start=1):
        parsed, errors = _parse_pref(raw, source_name=source_name)
        if errors:
            rejected_rows.append(_rejected(raw, line_no, errors, source_name, "dpo"))
        else:
            valid_rows.append(parsed)

    if fail_on_invalid and rejected_rows:
        raise ValueError(f"Found {len(rejected_rows)} invalid preference rows.")

    splits = _split(
        valid_rows,
        train=float(split_cfg["train"]),
        dev=float(split_cfg["dev"]),
        test=float(split_cfg["test"]),
        seed=int(split_cfg["seed"]),
    )
    write_jsonl(output_dir / "dpo_all.jsonl", valid_rows)
    write_jsonl(output_dir / "dpo_train.jsonl", splits["train"])
    write_jsonl(output_dir / "dpo_dev.jsonl", splits["dev"])
    write_jsonl(output_dir / "dpo_test.jsonl", splits["test"])
    write_jsonl(rejected_path, rejected_rows)

    quality = _quality_report(len(raw_rows), valid_rows, rejected_rows)
    write_json(quality_path, quality)
    _write_dataset_info(dataset_info_path)

    return {
        "dataset": "dpo",
        "input": str(input_path),
        "total_raw": len(raw_rows),
        "valid": len(valid_rows),
        "rejected": len(rejected_rows),
        "train": len(splits["train"]),
        "dev": len(splits["dev"]),
        "test": len(splits["test"]),
        "quality_report": str(quality_path),
    }
