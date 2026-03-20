from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import re
import unicodedata
from typing import Any, Mapping

from align.io import read_jsonl, write_json, write_jsonl
from align.prompts import SYSTEM_PROMPT


CATEGORY_LABELS = {
    "returns_refunds": "退货退款",
    "shipping_logistics": "物流配送",
    "product_consultation": "商品咨询",
    "order_modification": "订单修改",
    "warranty_repair": "售后维修",
    "complaint_resolution": "投诉处理",
    "general": "通用客服",
}
FALLBACK_CATEGORY = CATEGORY_LABELS["general"]
FAQ_DEFAULT_CATEGORY = CATEGORY_LABELS["product_consultation"]
CATEGORIES = list(CATEGORY_LABELS.values())

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
    "returns_refunds": CATEGORY_LABELS["returns_refunds"],
    "returns": CATEGORY_LABELS["returns_refunds"],
    "refund": CATEGORY_LABELS["returns_refunds"],
    "refunds": CATEGORY_LABELS["returns_refunds"],
    "return": CATEGORY_LABELS["returns_refunds"],
    "return_refund": CATEGORY_LABELS["returns_refunds"],
    "shipping_logistics": CATEGORY_LABELS["shipping_logistics"],
    "shipping": CATEGORY_LABELS["shipping_logistics"],
    "logistics": CATEGORY_LABELS["shipping_logistics"],
    "delivery": CATEGORY_LABELS["shipping_logistics"],
    "product_specs": CATEGORY_LABELS["product_consultation"],
    "product_spec": CATEGORY_LABELS["product_consultation"],
    "specs": CATEGORY_LABELS["product_consultation"],
    "product": CATEGORY_LABELS["product_consultation"],
    "product_consultation": CATEGORY_LABELS["product_consultation"],
    "consultation": CATEGORY_LABELS["product_consultation"],
    "order_modification": CATEGORY_LABELS["order_modification"],
    "order_change": CATEGORY_LABELS["order_modification"],
    "change_order": CATEGORY_LABELS["order_modification"],
    "modify_order": CATEGORY_LABELS["order_modification"],
    "after_sales": CATEGORY_LABELS["warranty_repair"],
    "aftersales": CATEGORY_LABELS["warranty_repair"],
    "post_sale": CATEGORY_LABELS["warranty_repair"],
    "warranty": CATEGORY_LABELS["warranty_repair"],
    "warranty_repair": CATEGORY_LABELS["warranty_repair"],
    "complaint_soothing": CATEGORY_LABELS["complaint_resolution"],
    "complaint": CATEGORY_LABELS["complaint_resolution"],
    "deescalation": CATEGORY_LABELS["complaint_resolution"],
    "complaint_resolution": CATEGORY_LABELS["complaint_resolution"],
    "general": CATEGORY_LABELS["general"],
    "other": CATEGORY_LABELS["general"],
    "others": CATEGORY_LABELS["general"],
    "misc": CATEGORY_LABELS["general"],
    "miscellaneous": CATEGORY_LABELS["general"],
    "退货退款": CATEGORY_LABELS["returns_refunds"],
    "退款": CATEGORY_LABELS["returns_refunds"],
    "退货": CATEGORY_LABELS["returns_refunds"],
    "退換": CATEGORY_LABELS["returns_refunds"],
    "退换": CATEGORY_LABELS["returns_refunds"],
    "物流配送": CATEGORY_LABELS["shipping_logistics"],
    "物流": CATEGORY_LABELS["shipping_logistics"],
    "配送": CATEGORY_LABELS["shipping_logistics"],
    "发货": CATEGORY_LABELS["shipping_logistics"],
    "快递": CATEGORY_LABELS["shipping_logistics"],
    "商品咨询": CATEGORY_LABELS["product_consultation"],
    "商品问题": CATEGORY_LABELS["product_consultation"],
    "規格": CATEGORY_LABELS["product_consultation"],
    "规格": CATEGORY_LABELS["product_consultation"],
    "咨询": CATEGORY_LABELS["product_consultation"],
    "尺寸": CATEGORY_LABELS["product_consultation"],
    "材质": CATEGORY_LABELS["product_consultation"],
    "颜色": CATEGORY_LABELS["product_consultation"],
    "顏色": CATEGORY_LABELS["product_consultation"],
    "库存": CATEGORY_LABELS["product_consultation"],
    "订单修改": CATEGORY_LABELS["order_modification"],
    "改地址": CATEGORY_LABELS["order_modification"],
    "修改订单": CATEGORY_LABELS["order_modification"],
    "取消订单": CATEGORY_LABELS["order_modification"],
    "售后维修": CATEGORY_LABELS["warranty_repair"],
    "售后": CATEGORY_LABELS["warranty_repair"],
    "维修": CATEGORY_LABELS["warranty_repair"],
    "保修": CATEGORY_LABELS["warranty_repair"],
    "投诉处理": CATEGORY_LABELS["complaint_resolution"],
    "投诉": CATEGORY_LABELS["complaint_resolution"],
    "差评": CATEGORY_LABELS["complaint_resolution"],
    "不满意": CATEGORY_LABELS["complaint_resolution"],
    "通用客服": CATEGORY_LABELS["general"],
    "其他": CATEGORY_LABELS["general"],
}

CATEGORY_INFERENCE_RULES: list[tuple[list[str], str]] = [
    (
        ["refund", "return", "rma", "退款", "退货", "退换", "換貨", "补偿", "价保", "仅退款", "退款申请"],
        CATEGORY_LABELS["returns_refunds"],
    ),
    (
        [
            "shipping",
            "delivery",
            "logistics",
            "tracking",
            "物流",
            "发货",
            "快递",
            "配送",
            "运单",
            "签收",
            "催单",
            "派送",
        ],
        CATEGORY_LABELS["shipping_logistics"],
    ),
    (
        [
            "spec",
            "size",
            "material",
            "compatib",
            "规格",
            "尺寸",
            "材质",
            "颜色",
            "顏色",
            "型号",
            "参数",
            "兼容",
            "咨询",
            "库存",
        ],
        CATEGORY_LABELS["product_consultation"],
    ),
    (
        [
            "change",
            "modify",
            "cancel",
            "address",
            "修改",
            "取消",
            "改地址",
            "改订单",
            "收货地址",
            "订单信息",
            "取消订单",
        ],
        CATEGORY_LABELS["order_modification"],
    ),
    (
        ["warranty", "repair", "after-sales", "aftersales", "保修", "维修", "售后", "故障", "坏了", "换新", "返修"],
        CATEGORY_LABELS["warranty_repair"],
    ),
    (
        ["complaint", "angry", "frustrated", "escalat", "投诉", "生气", "不满", "人工", "升级处理", "差评", "处理"],
        CATEGORY_LABELS["complaint_resolution"],
    ),
]

LEGACY_EN_SYSTEM_PROMPT_SIGNATURES = {
    "youareaprofessionalecommercecustomersupportassistant",
    "youareaecommercecustomersupportassistant",
}

TEMPLATE_NOISE_PATTERNS = (
    re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL),
    re.compile(r"#\s*e-[^\s,，。!?？!;；]*", flags=re.IGNORECASE),
    re.compile(r"#\s*e-s\[[^\]]*\]", flags=re.IGNORECASE),
    re.compile(r"&nbsp;", flags=re.IGNORECASE),
    re.compile(r"\[[^\]]*[xX][^\]]*\]"),
    re.compile(
        r"\[(?:数字|姓名|电话|手机号|地址|日期|时间|金额|站点|组织机构|机构|订单|快递|Name|name|ORDERID|EMAIL|PHONE|ID)[^\]]*\]"
    ),
)

PLACEHOLDER_SIGNATURE_PATTERNS = (
    re.compile(r"#\s*e-[^\s,，。!?？!;；]*", flags=re.IGNORECASE),
    re.compile(r"#\s*e-s\[[^\]]*\]", flags=re.IGNORECASE),
    re.compile(r"\[[^\]]*[xX][^\]]*\]"),
    re.compile(
        r"\[(?:数字|姓名|电话|手机号|地址|日期|时间|金额|站点|组织机构|机构|订单|快递|Name|name|ORDERID|EMAIL|PHONE|ID)[^\]]*\]"
    ),
)

GENERIC_CLOSING_PHRASES = {
    "请问还有其他可以帮到您的吗",
    "请问还有其他还可以帮到您的吗",
    "请问还有什么可以帮您",
    "还有其他问题吗",
    "还有什么问题吗",
    "感谢您的咨询",
    "感谢您对京东的支持，祝您生活愉快，再见",
    "祝您生活愉快",
    "如有问题随时联系",
    "谢谢您的支持",
    "后期有问题再来咨询妹子哦",
    "没有其他问题妹子就不打扰您了哦",
    "没有其他问题的话妹子就和您说再见了哦",
}

LOW_INFORMATION_RESPONSES = {
    "好的",
    "好",
    "好的呢",
    "好的哦",
    "嗯",
    "嗯呢",
    "恩",
    "知道了",
    "收到",
    "可以",
    "在的",
    "没问题",
    "是的",
    "是的哦",
    "是的呢",
    "好滴",
    "好的亲",
    "谢谢",
    "谢谢您",
    "新年快乐",
    "不客气",
    "不客气的",
    "您客气了",
    "您客气了哈",
    "麻烦您了",
    "辛苦您等待下哦",
}
BUSINESS_SIGNAL_KEYWORDS = [
    "订单",
    "退款",
    "退货",
    "物流",
    "发货",
    "配送",
    "地址",
    "售后",
    "维修",
    "投诉",
    "商品",
    "规格",
    "库存",
    "申请",
    "核实",
    "提供",
    "处理",
    "照片",
    "工单",
    "进度",
    "补发",
    "order",
    "refund",
    "return",
    "shipping",
    "delivery",
]

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


def _compact_text(value: str) -> str:
    compact: list[str] = []
    for ch in value.lower():
        if ch.isspace():
            continue
        category = unicodedata.category(ch)
        if category and category[0] in {"P", "S"}:
            continue
        compact.append(ch)
    return "".join(compact)


GENERIC_CLOSING_SIGNATURES = {_compact_text(text) for text in GENERIC_CLOSING_PHRASES}
LOW_INFORMATION_SIGNATURES = {_compact_text(text) for text in LOW_INFORMATION_RESPONSES}
GENERIC_CLOSING_REGEXES = (
    re.compile(r"^请问还有.*帮.*您.*吗[?？]*$"),
    re.compile(r"^还有.*问题.*吗[?？]*$"),
    re.compile(r"^感谢.*支持.*(再见|愉快).*$"),
    re.compile(r"^后期有问题再来咨询.*$"),
    re.compile(r"^没有.*问题.*(不打扰|再见).*$"),
    re.compile(r"^如有问题.*联系.*$"),
)
COURTESY_TEMPLATE_REGEXES = (
    re.compile(r"^很高兴遇到您.*帮到您.*吗[?？]*$"),
    re.compile(r"^您.*客气.*$"),
    re.compile(r"^不客气.*$"),
    re.compile(r"^缘聚缘散缘如水.*$"),
    re.compile(r"^还辛苦您.*评价.*$"),
    re.compile(r"^妹子.*评价.*$"),
    re.compile(r"^遇到像您这样.*评价.*$"),
    re.compile(r"^祝您[:：]?.*(开心|愉快).*$"),
)


def _strip_template_noise(value: str) -> str:
    text = _text(value)
    for pattern in TEMPLATE_NOISE_PATTERNS:
        text = pattern.sub(" ", text)
    return _text(text)


def _strip_jddc_placeholders(value: str) -> str:
    return _strip_template_noise(value)


def _clean_text(value: Any) -> str:
    return _strip_template_noise(_text(value))


def _has_placeholder_signature(value: str) -> bool:
    text = _text(value)
    return any(pattern.search(text) for pattern in PLACEHOLDER_SIGNATURE_PATTERNS)


def _is_generic_closing_only(value: str) -> bool:
    compact = _compact_text(value)
    if not compact:
        return False
    if compact in GENERIC_CLOSING_SIGNATURES:
        return True
    text = _clean_text(value)
    if any(pattern.match(text) for pattern in GENERIC_CLOSING_REGEXES):
        return True
    if any(pattern.match(text) for pattern in COURTESY_TEMPLATE_REGEXES):
        return True
    if "评价" in text and not _has_business_signal(text):
        return True
    if not _has_business_signal(text) and any(token in text for token in ["客气", "愉快", "开心", "谢谢", "亲爱", "妹子"]):
        return True
    return False


def _is_low_information_response(value: str) -> bool:
    compact = _compact_text(value)
    if not compact:
        return False
    if compact in LOW_INFORMATION_SIGNATURES:
        return True
    return len(compact) <= 6 and not _has_business_signal(value)


def _has_business_signal(value: str) -> bool:
    text = _clean_text(value).lower()
    return any(keyword in text for keyword in BUSINESS_SIGNAL_KEYWORDS)


def _resolve_system_prompt(value: Any) -> str:
    prompt = _clean_text(value)
    if not _is_usable_text(prompt):
        return SYSTEM_PROMPT
    if _compact_text(prompt) in LEGACY_EN_SYSTEM_PROMPT_SIGNATURES:
        return SYSTEM_PROMPT
    return prompt


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
    if token in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[token]
    for alias, label in CATEGORY_ALIASES.items():
        if alias and alias in token:
            return label
    return ""


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
    text = _clean_text(value).lower()
    for keywords, category in CATEGORY_INFERENCE_RULES:
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
        value = _pick(turn, ["text", "content", "utterance", "sentence", "msg", "message"])
        return _clean_text(value) if isinstance(value, str) else ""
    if isinstance(turn, list) and len(turn) >= 2:
        return _clean_text(turn[1]) if isinstance(turn[1], str) else ""
    if isinstance(turn, str):
        return _clean_text(turn)
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

    query = _clean_text(query)
    input_text = _clean_text(input_text)
    response = _clean_text(response)

    category_value = _pick_text(raw, ["category", "intent", "topic", "domain", "scene", "label"])
    category = _category(category_value)
    if not category:
        category = _infer_category_from_text(f"{query}\n{input_text}\n{response}")
    if not category and source_format == "faq":
        category = FAQ_DEFAULT_CATEGORY
    if not category:
        # Keep recoverable rows instead of dropping Chinese JDDC records on taxonomy misses.
        category = FALLBACK_CATEGORY

    source_id = _text(_pick(raw, ["source_id", "original_id", "ticket_id", "dialog_id", "session_id"])) or record_id
    return {
        "id": record_id or f"{source_format}_{line_no:08d}",
        "category": category,
        "system": _resolve_system_prompt(_pick(raw, ["system", "system_prompt", "sys_prompt"], SYSTEM_PROMPT)),
        "query": query,
        "input": input_text,
        "response": response,
        "source": _text(_pick(raw, ["source", "source_name"], source_name)) or source_name,
        "source_id": source_id,
    }


def _parse_sft(raw: Mapping[str, Any], source_name: str) -> tuple[dict[str, Any], list[str]]:
    raw_instruction = _pick(raw, SFT_KEYS["instruction"])
    raw_input = _pick(raw, SFT_KEYS["input"], "")
    raw_output = _pick(raw, SFT_KEYS["output"])
    instruction = _clean_text(raw_instruction)
    input_text = _clean_text(raw_input)
    output = _clean_text(raw_output)
    category = _category(_pick(raw, SFT_KEYS["category"]))
    if not category:
        category = _infer_category_from_text(f"{instruction}\n{input_text}\n{output}")
    if not category:
        category = FALLBACK_CATEGORY

    record = {
        "id": _text(_pick(raw, SFT_KEYS["id"])),
        "category": category,
        "system": _resolve_system_prompt(_pick(raw, SFT_KEYS["system"], SYSTEM_PROMPT)),
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "source": _text(_pick(raw, SFT_KEYS["source"], source_name)) or source_name,
        "source_id": _text(_pick(raw, SFT_KEYS["source_id"], "")),
    }
    errors = _require_text(record, ["id", "category", "system", "instruction", "input", "output"], {"input"})
    if record["category"] not in CATEGORIES:
        errors.append(f"category: unsupported `{record['category']}`")
    if not _is_usable_text(record["instruction"]):
        errors.append("instruction: must be substantive text")
    if not _is_usable_text(record["output"]):
        errors.append("output: must be substantive text")
    if _has_placeholder_signature(_text(raw_output)) and not _is_usable_text(record["output"]):
        errors.append("output: placeholder/template garbage")
    if _is_generic_closing_only(record["output"]):
        errors.append("output: generic closing only")
    if _is_low_information_response(record["output"]):
        errors.append("output: low-information response")
    return record, errors


def _parse_pref(raw: Mapping[str, Any], source_name: str) -> tuple[dict[str, Any], list[str]]:
    raw_prompt = _pick(raw, PREF_KEYS["prompt"])
    raw_chosen = _pick(raw, PREF_KEYS["chosen"])
    raw_rejected = _pick(raw, PREF_KEYS["rejected"])
    prompt = _clean_text(raw_prompt)
    chosen = _clean_text(raw_chosen)
    rejected = _clean_text(raw_rejected)
    category = _category(_pick(raw, PREF_KEYS["category"]))
    if not category:
        category = _infer_category_from_text(f"{prompt}\n{chosen}\n{rejected}")
    if not category:
        category = FALLBACK_CATEGORY

    record = {
        "id": _text(_pick(raw, PREF_KEYS["id"])),
        "category": category,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": _text(_pick(raw, PREF_KEYS["source"], source_name)) or source_name,
        "source_id": _text(_pick(raw, PREF_KEYS["source_id"], "")),
    }
    errors = _require_text(record, ["id", "category", "prompt", "chosen", "rejected"])
    if record["category"] not in CATEGORIES:
        errors.append(f"category: unsupported `{record['category']}`")
    if not _is_usable_text(record["prompt"]):
        errors.append("prompt: must be substantive text")
    if not _is_usable_text(record["chosen"]):
        errors.append("chosen: must be substantive text")
    if not _is_usable_text(record["rejected"]):
        errors.append("rejected: must be substantive text")
    if _has_placeholder_signature(_text(raw_chosen)) and not _is_usable_text(record["chosen"]):
        errors.append("chosen: placeholder/template garbage")
    if _is_generic_closing_only(record["chosen"]):
        errors.append("chosen: generic closing only")
    if _is_low_information_response(record["chosen"]):
        errors.append("chosen: low-information response")
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
