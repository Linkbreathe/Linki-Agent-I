#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

payload = json.load(sys.stdin)
tool_input = payload.get("tool_input") or {}
command = str(tool_input.get("command") or "")

if "rm -rf" in command or "git push --force" in command:
    print(json.dumps({"decision": "ask", "reason": "destructive command"}))
