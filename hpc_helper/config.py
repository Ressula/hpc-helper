from __future__ import annotations

import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore
    except ImportError:
        import tomli as tomllib  # type: ignore

CONFIG_DIR = Path.home() / ".hpc-helper"
CONFIG_FILE = CONFIG_DIR / "config.toml"


@dataclass
class Config:
    host: str
    user: str
    remote_home: str
    conda_env: str
    cpus: int = 4
    gpus: int = 1
    walltime: int = 200

    @classmethod
    def load(cls) -> Config:
        if not CONFIG_FILE.exists():
            raise FileNotFoundError("No config found. Run `hpc init` first.")
        with open(CONFIG_FILE, "rb") as f:
            data = tomllib.load(f)
        return cls(**data)

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for f in fields(self):
            v = getattr(self, f.name)
            lines.append(f'{f.name} = "{v}"' if isinstance(v, str) else f"{f.name} = {v}")
        CONFIG_FILE.write_text("\n".join(lines) + "\n")

    def conda_bin(self) -> str:
        return f"{self.remote_home}/miniconda3/bin/conda"

    def conda_prefix(self, env: Optional[str] = None) -> str:
        return f"{self.conda_bin()} run -n {env or self.conda_env}"
