from __future__ import annotations

import fnmatch
import shlex
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Callable, List, Optional, Tuple


def _load_hpcignore(directory: str) -> List[str]:
    """Return glob patterns from .hpcignore in the given directory."""
    p = Path(directory) / ".hpcignore"
    if not p.exists():
        return []
    patterns = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


def _tar_filter(patterns: List[str]) -> Callable[[tarfile.TarInfo], Optional[tarfile.TarInfo]]:
    """Return a tarfile filter function that excludes paths matching any pattern."""
    def _filter(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        # info.name is like "./subdir/file.py" — strip the leading ./
        rel = info.name.lstrip("./")
        if not rel:
            return info
        parts = rel.split("/")
        for pattern in patterns:
            # Match against the full relative path or any path component
            if any(fnmatch.fnmatch(p, pattern) for p in parts):
                return None
            if fnmatch.fnmatch(rel, pattern):
                return None
        return info
    return _filter


def ssh_run(host: str, command: str) -> int:
    """Run a remote command, streaming output to the terminal."""
    return subprocess.run(["ssh", "-t", host, command]).returncode


def ssh_capture(host: str, command: str) -> Tuple[int, str]:
    """Run a remote command and capture its stdout."""
    result = subprocess.run(
        ["ssh", host, command],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout


def ssh_upload_text(host: str, remote_path: str, content: str) -> int:
    """Write string content to a remote file via ssh + cat.

    Encodes as bytes to prevent Windows from converting \\n to \\r\\n,
    which sbatch rejects with a DOS line-break error.
    """
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    result = subprocess.run(
        ["ssh", host, f"cat > {remote_path}"],
        input=normalized.encode("utf-8"),
    )
    return result.returncode


def tar_push(local: str, host: str, remote_dest: str) -> int:
    """Push local directory contents to remote via Python tarfile + SSH.

    Uses Python's built-in tarfile module (PAX format) instead of the system
    tar binary, so Unicode/Chinese filenames work on every platform without
    relying on locale settings or Windows bsdtar quirks.
    """
    local_abs = str(Path(local).resolve())
    patterns = _load_hpcignore(local_abs)
    tar_filter = _tar_filter(patterns) if patterns else None

    remote_cmd = (
        f"mkdir -p {shlex.quote(remote_dest)} && "
        f"LC_ALL=C.UTF-8 tar xzf - -C {shlex.quote(remote_dest)}"
    )
    ssh = subprocess.Popen(["ssh", host, remote_cmd], stdin=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=ssh.stdin, mode="w|gz", format=tarfile.PAX_FORMAT) as tf:
            tf.add(local_abs, arcname=".", filter=tar_filter)
    finally:
        ssh.stdin.close()
    ssh.wait()
    return ssh.returncode


def tar_pull(
    host: str,
    remote_src: str,
    local_dest: str,
    exclude: Optional[List[str]] = None,
) -> int:
    """Pull a remote directory to local via tar-over-SSH + Python tarfile.

    Packs the directory *itself* on the remote side so the result under
    local_dest mirrors the remote layout:
        remote: .../results/     →  local: local_dest/results/

    Exclusion patterns are applied on the remote before archiving.
    Python tarfile handles extraction so Unicode filenames work on Windows too.
    """
    remote_path = PurePosixPath(remote_src.rstrip("/"))
    remote_parent = shlex.quote(str(remote_path.parent))
    remote_name = shlex.quote(remote_path.name)
    excl = " ".join(f"--exclude={shlex.quote(p)}" for p in (exclude or []))
    remote_cmd = (
        f"LC_ALL=C.UTF-8 tar --format=pax czf - {excl} -C {remote_parent} {remote_name}"
    )

    local_abs = str(Path(local_dest).resolve())
    Path(local_dest).mkdir(parents=True, exist_ok=True)

    ssh = subprocess.Popen(["ssh", host, remote_cmd], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=ssh.stdout, mode="r|gz") as tf:
        tf.extractall(path=local_abs)
    ssh.wait()
    return ssh.returncode
