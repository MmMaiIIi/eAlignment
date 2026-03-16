def test_import_utils() -> None:
    import src.utils.config  # noqa: F401
    import src.utils.jsonl  # noqa: F401
    import src.utils.llamafactory  # noqa: F401
    import src.utils.paths  # noqa: F401


def test_import_eval_rules() -> None:
    import src.eval.proxy_rules  # noqa: F401


def test_import_data_modules() -> None:
    import src.data.constants  # noqa: F401
    import src.data.normalization  # noqa: F401
    import src.data.parsers  # noqa: F401
    import src.data.pipeline  # noqa: F401
    import src.data.schemas  # noqa: F401
    import src.data.splitting  # noqa: F401
