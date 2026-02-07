from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def load_json_config(default_path: str | Path, config_path: str | Path | None = None) -> Tuple[Dict[str, Any], Path]:
    target = resolve_path(config_path or default_path)
    with target.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config, target


def dump_json_config(config: Dict[str, Any], output_path: str | Path) -> Path:
    target = resolve_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return target
