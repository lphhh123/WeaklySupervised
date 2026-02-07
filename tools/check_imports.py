from __future__ import annotations

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ_FILE = ROOT / "requirements.txt"

NAME_MAP = {
    "scikit-learn": "sklearn",
    "mamba-ssm": "mamba_ssm",
    "pyyaml": "yaml",
}


def iter_requirement_names(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split(";")[0].split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0]
        name = name.strip()
        if name:
            names.append(NAME_MAP.get(name, name))
    return names


def main() -> None:
    for module_name in iter_requirement_names(REQ_FILE):
        importlib.import_module(module_name)
    print("All requirements imported successfully.")


if __name__ == "__main__":
    main()
