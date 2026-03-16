import importlib


def test_repo_imports() -> None:
    modules = [
        "src",
        "src.config",
        "src.data",
        "src.data.io",
        "src.training",
        "src.eval",
        "src.utils",
        "src.utils.config_loader",
    ]
    for module in modules:
        importlib.import_module(module)

