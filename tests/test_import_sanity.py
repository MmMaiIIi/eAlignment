def test_import_utils() -> None:
    import src.utils.config  # noqa: F401
    import src.utils.jsonl  # noqa: F401
    import src.utils.paths  # noqa: F401


def test_import_eval_rules() -> None:
    import src.eval.proxy_rules  # noqa: F401
