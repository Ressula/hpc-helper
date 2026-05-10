from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Union

import yaml

from .config import Config

_SBATCH_HEADER = """\
#!/bin/bash
#SBATCH -A stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --job-name={name}
#SBATCH --nodes=1
#SBATCH -c {cpus}
#SBATCH --time={walltime}
#SBATCH --gres=gpu:{gpus}
#SBATCH --output={remote_home}/slurm-%j-{name}.out
"""


@dataclass
class RunStep:
    command: str  # fully expanded shell command, ready to exec on the compute node


@dataclass
class Group:
    name: str
    runs: List[RunStep]
    cpus: int
    gpus: int
    walltime: int


def _expand_run(item: Any, cfg: Config, remote_project: str, env: str) -> RunStep:
    """Turn a YAML run item into an executable shell command."""
    if isinstance(item, str):
        return RunStep(f"{cfg.conda_prefix(env)} python {remote_project}/{item}")
    if isinstance(item, dict):
        if "raw" in item:
            return RunStep(item["raw"])
        script = item["script"]
        item_env = item.get("env", env)
        return RunStep(f"{cfg.conda_prefix(item_env)} python {remote_project}/{script}")
    raise ValueError(f"Invalid run entry: {item!r}")


def parse_batch_file(
    path: Union[str, Path],
    cfg: Config,
    remote_project: str,
) -> List[Group]:
    with open(path) as f:
        data = yaml.safe_load(f)

    res = data.get("resources", {})
    default_cpus = res.get("cpus", cfg.cpus)
    default_gpus = res.get("gpus", cfg.gpus)
    default_walltime = res.get("time", cfg.walltime)
    default_env = res.get("env", cfg.conda_env)

    groups: List[Group] = []
    for g in data["groups"]:
        g_res = g.get("resources", {})
        env = g_res.get("env", default_env)
        runs = [_expand_run(r, cfg, remote_project, env) for r in g["runs"]]
        groups.append(Group(
            name=g["name"],
            runs=runs,
            cpus=g_res.get("cpus", default_cpus),
            gpus=g_res.get("gpus", default_gpus),
            walltime=g_res.get("time", default_walltime),
        ))
    return groups


def render_sbatch_script(group: Group, cfg: Config) -> str:
    header = _SBATCH_HEADER.format(
        name=group.name,
        cpus=group.cpus,
        gpus=group.gpus,
        walltime=group.walltime,
        remote_home=cfg.remote_home,
    )
    body = "\n".join(step.command for step in group.runs)
    return header + "\n" + body + "\n"
