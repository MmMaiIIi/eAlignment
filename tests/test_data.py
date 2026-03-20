import json
import subprocess
import sys
from pathlib import Path

from align.io import read_jsonl


def test_prepare_data_and_pref_smoke(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    interim = tmp_path / "interim"
    sft_rejected = interim / "sft_rejected.jsonl"
    dpo_rejected = interim / "dpo_rejected.jsonl"
    dpo_quality = interim / "dpo_quality_report.json"
    dataset_info = processed / "dataset_info.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--profile",
            "smoke",
            "--output-dir",
            str(processed),
            "--rejected-path",
            str(sft_rejected),
            "--dataset-info",
            str(dataset_info),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_pref.py",
            "--profile",
            "smoke",
            "--output-dir",
            str(processed),
            "--rejected-path",
            str(dpo_rejected),
            "--quality-path",
            str(dpo_quality),
            "--dataset-info",
            str(dataset_info),
        ],
        check=True,
    )

    sft_train = read_jsonl(processed / "sft_train.jsonl")
    dpo_train = read_jsonl(processed / "dpo_train.jsonl")
    sft_rej = read_jsonl(sft_rejected)
    dpo_rej = read_jsonl(dpo_rejected)
    quality = json.loads(dpo_quality.read_text(encoding="utf-8"))

    assert len(sft_train) >= 1
    assert len(dpo_train) >= 1
    assert len(sft_rej) >= 1
    assert len(dpo_rej) >= 1
    assert quality["dataset"] == "dpo"
    assert "issue_counts" in quality
    assert dataset_info.exists()


def test_prepare_data_external_source_formats(tmp_path: Path) -> None:
    samples = {
        "jddc": {
            "session_id": "jd_001",
            "dialog": [
                {"role": "user", "text": "Where is my order tracking update?"},
                {"role": "assistant", "text": "Please share your order number and I will check shipping status."},
            ],
        },
        "ecd": {
            "id": "ecd_001",
            "buyer_query": "I want to return this damaged item.",
            "seller_response": "Please share your order number and photos for return processing.",
            "context": "Order was delivered yesterday.",
        },
        "faq": {
            "faq_id": "faq_001",
            "question": "Is this bottle BPA free?",
            "answer": "Yes, the bottle material is BPA free.",
        },
    }
    required_keys = {"id", "category", "system", "instruction", "input", "output", "source", "source_id"}

    for source_format, row in samples.items():
        raw_path = tmp_path / f"{source_format}.jsonl"
        raw_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

        out_dir = tmp_path / source_format / "processed"
        rejected_path = tmp_path / source_format / "interim" / "sft_rejected.jsonl"
        dataset_info = out_dir / "dataset_info.json"

        subprocess.run(
            [
                sys.executable,
                "scripts/prepare_data.py",
                "--profile",
                "smoke",
                "--input",
                str(raw_path),
                "--output-dir",
                str(out_dir),
                "--rejected-path",
                str(rejected_path),
                "--dataset-info",
                str(dataset_info),
                "--source-format",
                source_format,
                "--fail-on-invalid",
            ],
            check=True,
        )

        sft_all = read_jsonl(out_dir / "sft_all.jsonl")
        sft_train = read_jsonl(out_dir / "sft_train.jsonl")
        rejected = read_jsonl(rejected_path)
        mapping = json.loads(dataset_info.read_text(encoding="utf-8"))

        assert len(sft_all) == 1
        assert len(sft_train) == 1
        assert len(rejected) == 0
        assert required_keys.issubset(sft_all[0].keys())
        assert sft_all[0]["category"]
        assert sft_all[0]["instruction"]
        assert sft_all[0]["output"]
        assert mapping["ecom_sft_seed"]["columns"] == {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system",
        }


def test_prepare_data_external_format_reports_normalization_errors(tmp_path: Path) -> None:
    raw_path = tmp_path / "faq_bad.jsonl"
    raw_path.write_text(json.dumps({"faq_id": "faq_bad_001", "question": "How to return?"}) + "\n", encoding="utf-8")

    out_dir = tmp_path / "processed"
    rejected_path = tmp_path / "interim" / "sft_rejected.jsonl"
    dataset_info = out_dir / "dataset_info.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--profile",
            "smoke",
            "--input",
            str(raw_path),
            "--output-dir",
            str(out_dir),
            "--rejected-path",
            str(rejected_path),
            "--dataset-info",
            str(dataset_info),
            "--source-format",
            "faq",
        ],
        check=True,
    )

    rejected = read_jsonl(rejected_path)
    assert len(rejected) == 1
    assert any("normalize: faq: missing response text" in err for err in rejected[0]["errors"])


def test_prepare_data_jddc_contract_categories_and_rejections(tmp_path: Path) -> None:
    rows = [
        {
            "session_id": "jd_ship_role",
            "dialog": [
                {"role": "user", "text": "Where is my shipping tracking update?"},
                {"role": "assistant", "text": "Please share your order id and I will check delivery progress."},
            ],
        },
        {
            "session_id": "jd_alias_refund",
            "dialog": [
                {"role": "buyer", "text": "I want a refund and return for this item."},
                {"role": "seller", "text": "I can help you start the return and refund process."},
            ],
        },
        {
            "session_id": "jd_list_specs",
            "dialog": [
                ["q", "What size and material is this jacket?"],
                ["a", "It is size M and cotton material."],
            ],
        },
        {
            "session_id": "jd_modify_context",
            "dialog": [
                {"role": "user", "text": "Hi"},
                {"role": "assistant", "text": "Hello, how can I help?"},
                {"role": "user", "text": "Please change my address."},
                {"role": "assistant", "text": "Sure, I can help modify the address."},
            ],
        },
        {
            "session_id": "jd_warranty",
            "dialog": [
                {"role": "human", "text": "My product is broken, can I get warranty repair?"},
                {"role": "bot", "text": "Yes, we can arrange after-sales repair support."},
            ],
        },
        {
            "session_id": "jd_complaint",
            "dialog": [
                {"role": "customer", "text": "I have a complaint and want to escalate this issue."},
                {"role": "agent", "text": "I am sorry you are frustrated and will help resolve this calmly."},
            ],
        },
        {
            "session_id": "jd_bad_no_pair",
            "dialog": [
                {"role": "assistant", "text": "Hello."},
            ],
        },
        {
            "session_id": "jd_bad_category",
            "dialog": [
                {"role": "user", "text": "hello there"},
                {"role": "assistant", "text": "ok noted"},
            ],
        },
    ]

    raw_path = tmp_path / "jddc_contract.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    out_dir = tmp_path / "processed"
    rejected_path = tmp_path / "interim" / "sft_rejected.jsonl"
    dataset_info = out_dir / "dataset_info.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--profile",
            "smoke",
            "--input",
            str(raw_path),
            "--output-dir",
            str(out_dir),
            "--rejected-path",
            str(rejected_path),
            "--dataset-info",
            str(dataset_info),
            "--source-format",
            "jddc",
            "--source-name",
            "jddc_contract",
        ],
        check=True,
    )

    sft_all = read_jsonl(out_dir / "sft_all.jsonl")
    sft_train = read_jsonl(out_dir / "sft_train.jsonl")
    sft_dev = read_jsonl(out_dir / "sft_dev.jsonl")
    sft_test = read_jsonl(out_dir / "sft_test.jsonl")
    rejected = read_jsonl(rejected_path)
    mapping = json.loads(dataset_info.read_text(encoding="utf-8"))

    assert len(sft_all) == 7
    assert len(rejected) == 1
    assert len(sft_train) + len(sft_dev) + len(sft_test) == len(sft_all)

    split_ids = {row["id"] for row in sft_train + sft_dev + sft_test}
    all_ids = {row["id"] for row in sft_all}
    assert split_ids == all_ids
    assert {row["id"] for row in sft_train}.isdisjoint({row["id"] for row in sft_dev})
    assert {row["id"] for row in sft_train}.isdisjoint({row["id"] for row in sft_test})
    assert {row["id"] for row in sft_dev}.isdisjoint({row["id"] for row in sft_test})

    by_id = {row["id"]: row for row in sft_all}
    assert by_id["jd_ship_role"]["category"] == "shipping_logistics"
    assert by_id["jd_alias_refund"]["category"] == "returns_refunds"
    assert by_id["jd_list_specs"]["category"] == "product_specs"
    assert by_id["jd_modify_context"]["category"] == "order_modification"
    assert by_id["jd_warranty"]["category"] == "after_sales"
    assert by_id["jd_complaint"]["category"] == "complaint_soothing"
    assert by_id["jd_bad_category"]["category"] == "general"

    assert by_id["jd_modify_context"]["instruction"] == "Please change my address."
    assert by_id["jd_modify_context"]["input"] == "user: Hi assistant: Hello, how can I help?"
    assert by_id["jd_modify_context"]["output"] == "Sure, I can help modify the address."

    required_keys = {"id", "category", "system", "instruction", "input", "output", "source", "source_id"}
    for row in sft_all:
        assert required_keys.issubset(row.keys())
        assert row["source"] == "jddc_contract"

    rejected_by_session = {row["raw"].get("session_id"): row for row in rejected}
    assert set(rejected_by_session) == {"jd_bad_no_pair"}
    assert any(
        "normalize: jddc: malformed dialogue (<2 usable turns)" in err
        for err in rejected_by_session["jd_bad_no_pair"]["errors"]
    )
    assert rejected_by_session["jd_bad_no_pair"]["dataset"] == "sft"
    assert rejected_by_session["jd_bad_no_pair"]["source"] == "jddc_contract"

    assert mapping["ecom_sft_seed"]["columns"] == {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "system": "system",
    }


def test_prepare_data_jddc_relaxed_chinese_acceptance_and_quality_guards(tmp_path: Path) -> None:
    rows = [
        {
            "session_id": "cn_refund",
            "dialog": [
                {"role": "user", "text": "这个商品可以退货退款吗？"},
                {"role": "assistant", "text": "可以的，这边帮您申请退款。"},
            ],
        },
        {
            "session_id": "cn_shipping",
            "dialog": [
                {"role": "buyer", "text": "我的快递到哪里了，还没发货吗？"},
                {"role": "seller", "text": "亲，这边帮您查询一下物流信息。"},
            ],
        },
        {
            "session_id": "cn_general_fallback",
            "dialog": [
                {"role": "user", "text": "你好，在吗"},
                {"role": "assistant", "text": "好的"},
            ],
        },
        {
            "session_id": "recover_pair_with_malformed_turns",
            "dialog": [
                {"role": "user", "text": "请帮我查一下物流"},
                {"role": "assistant", "text": 12345},
                {"speaker": "unknown", "content": "ignore-me"},
                {"role": "assistant", "text": "稍等"},
            ],
        },
        {
            "session_id": "short_reply_ok",
            "dialog": [
                ["q", "在吗"],
                ["a", "可以的"],
            ],
        },
        {
            "session_id": "bad_single_turn",
            "dialog": [
                {"role": "assistant", "text": "好的"},
            ],
        },
        {
            "session_id": "bad_punctuation_reply",
            "dialog": [
                {"role": "user", "text": "可以处理吗"},
                {"role": "assistant", "text": "..."},
            ],
        },
    ]

    raw_path = tmp_path / "jddc_relaxed.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    out_dir = tmp_path / "processed"
    rejected_path = tmp_path / "interim" / "sft_rejected.jsonl"
    dataset_info = out_dir / "dataset_info.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--profile",
            "smoke",
            "--input",
            str(raw_path),
            "--output-dir",
            str(out_dir),
            "--rejected-path",
            str(rejected_path),
            "--dataset-info",
            str(dataset_info),
            "--source-format",
            "jddc",
            "--source-name",
            "jddc_relaxed",
        ],
        check=True,
    )

    sft_all = read_jsonl(out_dir / "sft_all.jsonl")
    rejected = read_jsonl(rejected_path)
    by_id = {row["id"]: row for row in sft_all}

    assert len(sft_all) == 5
    assert len(rejected) == 2

    assert by_id["cn_refund"]["category"] == "returns_refunds"
    assert by_id["cn_shipping"]["category"] == "shipping_logistics"
    assert by_id["cn_general_fallback"]["category"] == "general"
    assert by_id["short_reply_ok"]["output"] == "可以的"
    assert by_id["recover_pair_with_malformed_turns"]["output"] == "稍等"

    rejected_by_session = {row["raw"].get("session_id"): row for row in rejected}
    assert set(rejected_by_session) == {"bad_single_turn", "bad_punctuation_reply"}
    assert any(
        "normalize: jddc: malformed dialogue (<2 usable turns)" in err
        for err in rejected_by_session["bad_single_turn"]["errors"]
    )
    assert any(
        "normalize: jddc: malformed dialogue (<2 usable turns)" in err
        for err in rejected_by_session["bad_punctuation_reply"]["errors"]
    )
