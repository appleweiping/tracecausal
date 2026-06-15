from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracecausal.contracts import TRACECAUSAL_CONTRACT, required_paths


def main() -> int:
    missing = [path for path in required_paths(ROOT) if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing required doc: {path}")
        return 1

    config = ROOT / "configs" / "experiments" / "formal_tracecausal.yaml"
    text = config.read_text(encoding="utf-8")
    missing_markers = [
        marker for marker in TRACECAUSAL_CONTRACT.required_config_markers if marker not in text
    ]
    if missing_markers:
        for marker in missing_markers:
            print(f"missing config marker: {marker}")
        return 1

    print("TraceCausal local contract validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

