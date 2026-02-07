import json
from pathlib import Path
from typing import Any, Dict, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["_config_path"] = str(config_path)
    return resolve_paths(config)


def resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    root = PROJECT_ROOT
    for key in ("paths", "path"):
        if key not in config:
            continue
        for path_key, value in list(config[key].items()):
            if not isinstance(value, str):
                continue
            path_value = Path(value)
            if not path_value.is_absolute():
                config[key][path_key] = str((root / path_value).resolve())
    return config


def merge_cli_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            continue
        key_path, value = item.split("=", 1)
        keys = key_path.split(".")
        cursor = config
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = _coerce_value(value)
    return config


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
