from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_jsonc(path: str | Path) -> dict[str, Any]:
    """Load JSON with // line comments (JSONC)."""
    text = Path(path).read_text(encoding="utf-8")
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return json.loads(text)
