"""Package init: load .env before any module reads the environment.

Every config constant in this package is resolved at import time
(`os.environ.get(...)` at module level), so a .env loaded later than this would
be loaded too late to matter. Doing it here means it happens before
`app.llm_backends` or `app.evidence` are imported, whatever imports them first.

The repo root is checked before `backend/`, and an existing real environment
variable always wins (`override=False`) so a container or CI secret is never
silently replaced by a stale file on disk.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is a convenience, not a requirement
    load_dotenv = None

if load_dotenv is not None and not os.environ.get("BIOORCH_SKIP_DOTENV"):
    _here = Path(__file__).resolve()
    for _candidate in (_here.parent.parent.parent / ".env",   # repo root
                       _here.parent.parent / ".env"):         # backend/
        if _candidate.is_file():
            load_dotenv(_candidate, override=False)
            break
