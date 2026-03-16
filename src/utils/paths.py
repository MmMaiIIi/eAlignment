from pathlib import Path


def repo_root() -> Path:
    """Return repository root based on this file location."""
    return Path(__file__).resolve().parents[2]


def from_root(*parts: str) -> Path:
    """Build an absolute path under repository root."""
    return repo_root().joinpath(*parts)
