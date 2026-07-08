#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys

payload = json.load(sys.stdin)
tool_input = payload.get("tool_input") or {}
file_path = str(tool_input.get("file_path") or "")

if re.search(r"(^|/)\.env($|\.)|(^|/)secrets/|\.pem$", file_path):
    print(f"secret-like path blocked: {file_path}", file=sys.stderr)
    raise SystemExit(2)
