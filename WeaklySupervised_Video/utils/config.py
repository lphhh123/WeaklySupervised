import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ConfigPaths:
    data_dir: str
    mapping_path: str | None
    output_dir: str
    ckpt_dir: str


def _resolve_path(root: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def load_config(config_path: str, repo_root: str) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    root = Path(repo_root)
    paths = cfg.get("paths", {})
    for key in ("data_dir", "mapping_path", "output_dir", "ckpt_dir"):
        if key in paths:
            paths[key] = _resolve_path(root, paths[key])
    cfg["paths"] = paths
    return cfg


def merge_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if value is None:
            continue
        cfg[key] = value
    return cfg
