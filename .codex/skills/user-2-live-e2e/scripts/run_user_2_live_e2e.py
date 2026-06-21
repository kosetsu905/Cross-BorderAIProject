from __future__ import annotations

from pathlib import Path
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from user_2_live_e2e_runner import main as _main


if __name__ == "__main__":
    raise SystemExit(_main())
