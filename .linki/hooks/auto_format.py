#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

payload = json.load(sys.stdin)
tool_input = payload.get("tool_input") or {}
file_path = str(tool_input.get("file_path") or "")
workspace = Path(str(payload.get("workspace") or "."))

if not file_path.endswith(".py"):
    raise SystemExit(0)

black = shutil.which("black")
if not black:
    raise SystemExit(0)

completed = subprocess.run(
    [black, "--quiet", file_path],
    cwd=workspace,
    text=True,
    capture_output=True,
    check=False,
)
if completed.returncode == 0:
    print("formatted with black")
else:
    sys.stderr.write(completed.stderr)
    raise SystemExit(completed.returncode)
