from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import json


@dataclass(frozen=True)
class GenerationRecord:
    prompt: str
    response: str
    code: str
    model: str
    wrapper_metadata: Dict[str, Any]
    seed: int
    normal_stats: Dict[str, float]
    thresholds: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: Optional[str] = None


def save_generation_record(record: GenerationRecord, root: Union[str, Path], *, name: str) -> Path:
    root = Path(root)
    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "policy.py").write_text(record.code, encoding="utf-8")
    (target / "prompt.txt").write_text(record.prompt, encoding="utf-8")
    (target / "response.txt").write_text(record.response, encoding="utf-8")
    (target / "metadata.json").write_text(
        json.dumps(asdict(record), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target
