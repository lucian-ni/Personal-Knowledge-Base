"""Cross-platform dev launcher for the FastAPI backend.

Run with::

    uv run python scripts/dev_api.py

This works the same on macOS, Linux, and Windows without needing the
``PYTHONPATH=...`` shell prefix (which differs between sh/bash/zsh, cmd.exe,
and PowerShell). It inserts both source roots onto ``sys.path`` and exports
``PYTHONPATH`` so uvicorn's auto-reloader child process can still resolve
``pkb_ingestion`` after a reload.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_PATHS = [
    ROOT / "apps" / "api" / "src",
    ROOT / "packages" / "ingestion" / "src",
]

for path in SRC_PATHS:
    sys.path.insert(0, str(path))

# Inheritable by the reloader's child process on any OS.
os.environ["PYTHONPATH"] = os.pathsep.join(
    [*map(str, SRC_PATHS), os.environ.get("PYTHONPATH", "")]
)

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        "pkb_api.main:app",
        host=os.environ.get("API_HOST", "127.0.0.1"),
        port=int(os.environ.get("API_PORT", "8000")),
        reload=True,
        app_dir=str(ROOT / "apps" / "api" / "src"),
    )
