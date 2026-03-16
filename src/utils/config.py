import json
from pathlib import Path
from typing import Any, Dict


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """
    Load YAML config with a fallback to JSON parsing.

    Stage 0 keeps configs YAML-compatible and lightweight.
    """
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"Config is not a mapping: {path}")
        return data
    except ModuleNotFoundError:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"Config is not a mapping: {path}")
        return data
