from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sys
import time
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
from rich.console import Console
from rich.table import Table

from .batch import parse_batch_file, render_sbatch_script
from .config import Config
from .remote import _load_hpcignore, is_ignored, tar_pull, tar_push, ssh_capture, ssh_run, ssh_upload_text
from .session import Session

console = Console()

_MANIFEST_DIR = Path.home() / ".hpc-helper" / "manifests"


def _manifest_path(local_abs: str, remote_dest: str) -> Path:
    key = hashlib.md5(f"{local_abs}||{remote_dest}".encode()).hexdigest()[:16]
    return _MANIFEST_DIR / f"{key}.json"


def _file_sig(p: Path) -> List[int]:
    st = p.stat()
    return [st.st_mtime_ns, st.st_size]


def _scan_files(local_abs: str, patterns: List[str]) -> Dict[str, List[int]]:
    """Walk local_abs, skip ignored paths, return {rel_path: [mtime_ns, size]}."""
    sigs: Dict[str, List[int]] = {}
    for dirpath, dirnames, filenames in os.walk(local_abs):
        rel_dir = os.path.relpath(dirpath, local_abs).replace("\\", "/")
        if rel_dir == ".":
            rel_dir = ""
        # Prune ignored dirs so os.walk never descends into them
        dirnames[:] = [
            d for d in sorted(dirnames)
            if not is_ignored(f"{rel_dir}/{d}".lstrip("/"), patterns)
        ]
        for fname in sorted(filenames):
            rel = f"{rel_dir}/{fname}".lstrip("/").replace("\\", "/")
            if not is_ignored(rel, patterns):
                sigs[rel] = _file_sig(Path(local_abs) / rel)
    return sigs

_ENTRY_SH = """\
#!/bin/bash
#SBATCH -A {account}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --job-name={name}
#SBATCH --nodes=1
#SBATCH -c {cpus}
#SBATCH --time={walltime}
#SBATCH --gres=gpu:{gpus}
sleep infinity
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_config() -> Config:
    try:
        return Config.load()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


def _require_job(session: Session) -> str:
    if not session.job_id:
        console.print("[red]No active job. Run `hpc up` first.[/red]")
        sys.exit(1)
    return session.job_id


def _require_project(session: Session) -> str:
    if not session.remote_project:
        console.print("[red]No remote project path set. Run `hpc push` first.[/red]")
        sys.exit(1)
    return session.remote_project


# ── CLI root ──────────────────────────────────────────────────────────────────

@click.group()
def cli() -> None:
    """hpc-helper — streamlined CLI for HPC Slurm workflows."""


# ── hpc init ──────────────────────────────────────────────────────────────────

@cli.command()
def init() -> None:
    """Interactive setup wizard. Run once to configure your cluster connection."""
    console.print("[bold]hpc-helper setup[/bold]\n")

    # Pre-fill defaults from existing config if present
    existing: dict = {}
    try:
        cfg = Config.load()
        existing = {f.name: getattr(cfg, f.name) for f in dc_fields(cfg)}
    except FileNotFoundError:
        pass

    def ask(label: str, key: str, fallback=None):
        default = existing.get(key, fallback)
        return click.prompt(label, default=str(default) if default is not None else "")

    host = ask("SSH alias (from ~/.ssh/config)", "host")
    user = ask("Cluster username", "user")
    remote_home = ask("Remote home directory", "remote_home", f"/home/scc/{user}")
    conda_env = ask("Default conda environment", "conda_env", "base")
    account = ask("Slurm account", "account", "stu")
    partition = ask("Slurm partition", "partition", "Students")
    qos = ask("Slurm QOS", "qos", "qos_stu_default")
    cpus = int(ask("Default CPUs per job", "cpus", 4))
    gpus = int(ask("Default GPUs per job", "gpus", 1))
    walltime = int(ask("Default wall-time (minutes)", "walltime", 200))

    Config(
        host=host,
        user=user,
        remote_home=remote_home,
        conda_env=conda_env,
        account=account,
        partition=partition,
        qos=qos,
        cpus=cpus,
        gpus=gpus,
        walltime=walltime,
    ).save()

    from .config import CONFIG_FILE
    console.print(f"\n[green]Config saved to {CONFIG_FILE}[/green]")


# ── hpc up ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--cpus", default=None, type=int, help="Override CPU count.")
@click.option("--gpus", default=None, type=int, help="Override GPU count.")
@click.option("--time", "walltime", default=None, type=int, help="Wall-time in minutes.")
@click.option("--name", default="hpc-session", show_default=True, help="Slurm job name.")
def up(cpus: Optional[int], gpus: Optional[int], walltime: Optional[int], name: str) -> None:
    """Allocate a GPU node (submit holder job and wait until running)."""
    cfg = _load_config()
    session = Session.load()

    if session.job_id:
        console.print(f"[yellow]Job {session.job_id} is already active. Run `hpc down` first.[/yellow]")
        sys.exit(1)

    script = _ENTRY_SH.format(
        account=cfg.account,
        partition=cfg.partition,
        qos=cfg.qos,
        name=name,
        cpus=cpus or cfg.cpus,
        gpus=gpus or cfg.gpus,
        walltime=walltime or cfg.walltime,
    )

    entry_path = f"{cfg.remote_home}/entry.sh"

    with console.status("Uploading entry.sh..."):
        rc = ssh_upload_text(cfg.host, entry_path, script)
    if rc != 0:
        console.print("[red]Failed to upload entry.sh.[/red]")
        sys.exit(1)

    with console.status("Submitting job..."):
        rc, out = ssh_capture(cfg.host, f"sbatch {entry_path}")
    if rc != 0:
        console.print(f"[red]sbatch failed:[/red]\n{out}")
        sys.exit(1)

    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        console.print(f"[red]Could not parse job ID from sbatch output:[/red] {out!r}")
        sys.exit(1)
    job_id = m.group(1)
    console.print(f"Submitted job [bold]{job_id}[/bold]")

    # Poll until the job enters RUNNING state
    with console.status(f"Waiting for job {job_id} to start...") as status:
        while True:
            rc, out = ssh_capture(cfg.host, f"squeue -h -j {job_id} -o '%T %N'")
            line = out.strip()
            if not line:
                console.print(f"[red]Job {job_id} disappeared from the queue (failed to allocate?).[/red]")
                sys.exit(1)
            parts = line.split()
            state = parts[0]
            if state == "RUNNING":
                node = parts[1] if len(parts) > 1 else "unknown"
                break
            if state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                console.print(f"[red]Job entered terminal state: {state}.[/red]")
                sys.exit(1)
            status.update(f"Waiting for job {job_id}... (state: {state})")
            time.sleep(4)

    session.job_id = job_id
    session.node = node
    session.save()
    console.print(f"[green]Job {job_id} is running on {node}.[/green]")


# ── hpc down ──────────────────────────────────────────────────────────────────

@cli.command()
def down() -> None:
    """Cancel the active job and release the GPU."""
    cfg = _load_config()
    session = Session.load()
    job_id = _require_job(session)

    with console.status(f"Cancelling job {job_id}..."):
        rc, out = ssh_capture(cfg.host, f"scancel {job_id}")
    if rc != 0:
        console.print(f"[red]scancel failed:[/red] {out}")
        sys.exit(1)

    session.clear_job()
    console.print(f"[green]Job {job_id} cancelled. GPU released.[/green]")


# ── hpc status ────────────────────────────────────────────────────────────────

@cli.command()
def status() -> None:
    """Show your jobs currently in the Slurm queue."""
    cfg = _load_config()
    _, out = ssh_capture(cfg.host, f"squeue -u {cfg.user}")
    console.print(out.strip() if out.strip() else "No jobs in queue.")


# ── hpc ps ────────────────────────────────────────────────────────────────────

@cli.command()
def ps() -> None:
    """Show the session state cached locally by hpc-helper."""
    cfg = _load_config()
    session = Session.load()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Host", cfg.host)
    table.add_row("User", cfg.user)
    job_str = (
        f"{session.job_id}  (on {session.node})"
        if session.job_id
        else "[dim]none[/dim]"
    )
    table.add_row("Active job", job_str)
    table.add_row("Remote project", session.remote_project or "[dim]none[/dim]")
    table.add_row("Conda env", cfg.conda_env)
    if session.batch_jobs:
        pairs = "  ".join(f"{k}={v}" for k, v in session.batch_jobs.items())
        table.add_row("Batch jobs", pairs)
    console.print(table)


# ── hpc push ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("local", default=".", required=False)
@click.option(
    "--to",
    "remote_subpath",
    default=None,
    metavar="REMOTE_PATH",
    help="Full remote destination path under remote_home (e.g. experiments/run_42).",
)
@click.option(
    "--full",
    is_flag=True,
    help="Force a full sync, ignoring the local change manifest.",
)
def push(local: str, remote_subpath: Optional[str], full: bool) -> None:
    """Sync a local directory to the cluster.

    Only files that changed since the last push are transferred (incremental).
    Use --full to force a complete re-sync.

    \b
    Examples:
      hpc push                            # ~/homework/project1 → remote:projects/project1/
      hpc push ./src                      # sync only the src/ subdirectory
      hpc push --to experiments/run_42   # ~/homework/project1 → remote:experiments/run_42/
      hpc push --full                     # re-send everything regardless of changes
    """
    cfg = _load_config()
    session = Session.load()

    local_path = Path(local).resolve()
    if not local_path.exists():
        console.print(f"[red]Local path does not exist: {local_path}[/red]")
        console.print("[dim]Tip: --to sets the REMOTE destination, not a local path.[/dim]")
        sys.exit(1)

    if remote_subpath:
        stripped = remote_subpath.lstrip("/")
        if stripped.startswith("~/"):
            stripped = stripped[2:]
        remote_project = f"{cfg.remote_home}/{stripped}"
    else:
        remote_project = f"{cfg.remote_home}/projects/{local_path.name}"

    ssh_capture(cfg.host, f"mkdir -p {remote_project}")

    # ── manifest-based incremental push ──────────────────────────────────────
    local_abs = str(local_path)
    patterns = _load_hpcignore(local_abs)
    manifest_file = _manifest_path(local_abs, remote_project)

    manifest: Dict[str, List[int]] = {}
    if not full and manifest_file.exists():
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    current_sigs = _scan_files(local_abs, patterns)

    if manifest:
        changed = [p for p, sig in current_sigs.items() if manifest.get(p) != sig]
        if not changed:
            console.print("[green]Nothing changed — skipping push.[/green]")
            session.remote_project = remote_project
            session.save()
            return
        console.print(
            f"Pushing [bold]{len(changed)}[/bold] changed file(s) "
            f"([dim]{len(current_sigs) - len(changed)} unchanged[/dim]) → "
            f"[bold]{cfg.host}:{remote_project}/[/bold]"
        )
        rc = tar_push(local_abs, cfg.host, remote_project, files=changed)
    else:
        console.print(
            f"Pushing [bold]{len(current_sigs)}[/bold] file(s) → "
            f"[bold]{cfg.host}:{remote_project}/[/bold]"
        )
        rc = tar_push(local_abs, cfg.host, remote_project, files=None)

    if rc != 0:
        console.print("[red]Push failed.[/red]")
        sys.exit(1)

    # Save updated manifest only after a successful transfer
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(current_sigs), encoding="utf-8")

    session.remote_project = remote_project
    session.save()
    console.print(f"[green]Done.[/green]")


# ── hpc pull ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("remote_subpath", default=None, required=False)
@click.option("--to", "local_dest", default=None, help="Local destination directory.")
def pull(remote_subpath: Optional[str], local_dest: Optional[str]) -> None:
    """Download files from the cluster. Slurm log files are always excluded.

    Without arguments, syncs the remote project into the parent of the current
    directory (git-pull style), so the project folder itself is updated in place.

    Pass a sub-path to download a specific subdirectory into the current directory.
    """
    cfg = _load_config()
    session = Session.load()

    if remote_subpath:
        if remote_subpath.startswith("/"):
            remote_src = remote_subpath
        else:
            base = session.remote_project or f"{cfg.remote_home}/projects"
            remote_src = f"{base}/{remote_subpath.rstrip('/')}"
        effective_dest = local_dest or "."
    elif session.remote_project:
        remote_src = session.remote_project
        # Default: unpack into parent so project1/ is updated in-place (like git pull)
        effective_dest = local_dest or ".."
    else:
        console.print("[red]No remote project path known. Run `hpc push` first or pass a path.[/red]")
        sys.exit(1)

    console.print(f"Pulling [bold]{cfg.host}:{remote_src}[/bold] → [bold]{effective_dest}[/bold]")
    rc = tar_pull(cfg.host, remote_src, effective_dest, exclude=["slurm-*.out", ".git"])
    if rc != 0:
        console.print("[red]Pull failed.[/red]")
        sys.exit(1)
    console.print("[green]Done.[/green]")


# ── hpc run ───────────────────────────────────────────────────────────────────

@cli.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("script", required=False)
@click.argument("script_args", nargs=-1, type=click.UNPROCESSED)
@click.option("--env", "conda_env", default=None, help="Override conda environment for this run.")
@click.option("--raw", default=None, metavar="CMD", help="Run a raw shell command instead of a Python script.")
@click.option("--no-conda", is_flag=True, help="Skip conda, use system python.")
def run(
    script: Optional[str],
    script_args: tuple,
    conda_env: Optional[str],
    raw: Optional[str],
    no_conda: bool,
) -> None:
    """Run a script on the active GPU node (blocking — waits until it finishes).

    Requires an active job (hpc up) and code on the cluster (hpc push).

    \b
    Examples:
      hpc run train.py --lr 1e-3 --epochs 100
      hpc run --env eval_env eval.py --checkpoint best.pt
      hpc run --no-conda python3 other_script.py
      hpc run --raw "bash cleanup.sh && echo done"
    """
    cfg = _load_config()
    session = Session.load()
    job_id = _require_job(session)

    if raw:
        # Wrap the raw command in bash -c so compound expressions work
        remote_cmd = f"srun --jobid={job_id} bash -c {shlex.quote(raw)}"
    else:
        if not script:
            console.print("[red]Provide a SCRIPT argument or use --raw CMD.[/red]")
            sys.exit(1)
        remote_project = _require_project(session)
        script_path = f"{remote_project}/{script}"
        args_str = " ".join(script_args)

        if no_conda:
            remote_cmd = f"srun --jobid={job_id} python {script_path} {args_str}"
        else:
            prefix = cfg.conda_prefix(conda_env)
            remote_cmd = f"srun --jobid={job_id} {prefix} python {script_path} {args_str}"

    console.print(f"[dim]{remote_cmd}[/dim]\n")
    rc = ssh_run(cfg.host, remote_cmd)
    if rc != 0:
        console.print(f"[red]Run exited with code {rc}.[/red]")
        sys.exit(rc)


# ── hpc shell ─────────────────────────────────────────────────────────────────

@cli.command()
def shell() -> None:
    """Open an interactive bash session on the active GPU node.

    Requires an active job (hpc up). Type `exit` to leave without cancelling the job.
    """
    cfg = _load_config()
    session = Session.load()
    job_id = _require_job(session)
    console.print(f"Attaching to job [bold]{job_id}[/bold] on [bold]{session.node}[/bold]...\n")
    ssh_run(cfg.host, f"srun --jobid={job_id} --overlap --pty bash")


# ── hpc logs ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--lines", default=50, show_default=True, help="Number of lines to show before following.")
@click.option("--job", "job_filter", default=None, metavar="JOBID", help="Job ID to tail.")
@click.option("--group", default=None, metavar="NAME", help="Batch group name to tail.")
def logs(lines: int, job_filter: Optional[str], group: Optional[str]) -> None:
    """Tail the Slurm output log for a batch job.

    Without arguments, tails the most recently submitted batch group.
    """
    cfg = _load_config()
    session = Session.load()

    if group:
        job_id = session.batch_jobs.get(group)
        if not job_id:
            console.print(f"[red]No batch job found for group '{group}'.[/red]")
            sys.exit(1)
        log_path = f"{cfg.remote_home}/slurm-{job_id}-{group}.out"
    elif job_filter:
        matching = [g for g, j in session.batch_jobs.items() if j == job_filter]
        name_suffix = f"-{matching[0]}" if matching else ""
        log_path = f"{cfg.remote_home}/slurm-{job_filter}{name_suffix}.out"
    elif session.batch_jobs:
        last_group, last_job = list(session.batch_jobs.items())[-1]
        log_path = f"{cfg.remote_home}/slurm-{last_job}-{last_group}.out"
    else:
        console.print("[red]No batch jobs in session. Submit with `hpc batch` first.[/red]")
        sys.exit(1)

    console.print(f"[dim]→ {cfg.host}:{log_path}[/dim]\n")
    ssh_run(cfg.host, f"tail -n {lines} -f {log_path}")


# ── hpc batch ─────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("batch_file")
def batch(batch_file: str) -> None:
    """Submit groups of runs defined in a YAML file.

    Each group becomes one sbatch job. Within a group, scripts run sequentially.
    All groups are queued at once; Slurm runs them one after another.

    Requires code on the cluster (hpc push) before submitting.
    """
    cfg = _load_config()
    session = Session.load()

    if not session.remote_project:
        console.print("[yellow]Warning: no remote project set. Run `hpc push` first.[/yellow]")

    remote_project = session.remote_project or f"{cfg.remote_home}/projects"
    groups = parse_batch_file(batch_file, cfg, remote_project)

    staging = f"{cfg.remote_home}/.hpc-helper-batch"
    ssh_capture(cfg.host, f"mkdir -p {staging}")

    submitted = 0
    for group in groups:
        script_content = render_sbatch_script(group, cfg)
        remote_script = f"{staging}/{group.name}.sh"

        with console.status(f"Uploading [bold]{group.name}[/bold]..."):
            rc = ssh_upload_text(cfg.host, remote_script, script_content)
        if rc != 0:
            console.print(f"[red]Failed to upload script for group '{group.name}'.[/red]")
            continue

        rc, out = ssh_capture(cfg.host, f"chmod +x {remote_script} && sbatch {remote_script}")
        if rc != 0:
            console.print(f"[red]sbatch failed for group '{group.name}':[/red] {out.strip()}")
            continue

        m = re.search(r"Submitted batch job (\d+)", out)
        if m:
            job_id = m.group(1)
            session.batch_jobs[group.name] = job_id
            console.print(f"  Queued [bold]{group.name}[/bold] → job [bold]{job_id}[/bold]")
            submitted += 1
        else:
            console.print(f"[yellow]Could not parse job ID for '{group.name}': {out!r}[/yellow]")

    session.save()
    console.print(f"\n[green]{submitted}/{len(groups)} group(s) queued.[/green] Use `hpc status` to monitor.")
