from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

SESSION_FILE = Path.home() / ".hpc-helper" / "session.json"


@dataclass
class Session:
    job_id: Optional[str] = None
    node: Optional[str] = None
    remote_project: Optional[str] = None
    # group_name -> job_id for batch-submitted jobs
    batch_jobs: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> Session:
        if not SESSION_FILE.exists():
            return cls()
        try:
            data = json.loads(SESSION_FILE.read_text())
            return cls(**data)
        except Exception:
            return cls()

    def save(self) -> None:
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        SESSION_FILE.write_text(json.dumps(asdict(self), indent=2))

    def clear_job(self) -> None:
        self.job_id = None
        self.node = None
        self.save()
