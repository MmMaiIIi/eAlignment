from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def load_yaml(path: str | Path) -> dict[str, Any]:
    file_path = resolve(path)
    text = file_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ModuleNotFoundError:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {file_path}")
    return data


def load_profile(profile: str | None = None) -> tuple[str, dict[str, Any]]:
    cfg = load_yaml("configs/profiles.yaml")
    profiles = cfg.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("configs/profiles.yaml must define a mapping at `profiles`.")
    selected = profile or str(cfg.get("default_profile", "smoke"))
    if selected not in profiles:
        raise ValueError(f"Unknown profile `{selected}`. Available: {sorted(profiles.keys())}")
    values = profiles[selected]
    if not isinstance(values, dict):
        raise ValueError(f"Profile `{selected}` must be a mapping.")
    return selected, values

