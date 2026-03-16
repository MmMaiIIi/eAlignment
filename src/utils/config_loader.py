from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a config file from YAML (or JSON-compatible YAML)."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    text = config_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Config file is empty: {config_path}")

    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {config_path}")
    return data

