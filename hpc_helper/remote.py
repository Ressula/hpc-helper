from __future__ import annotations

import subprocess
from typing import List, Optional, Tuple


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


def rsync_push(local: str, host: str, remote_dest: str) -> int:
    """Push local directory contents into remote_dest via rsync.

    Trailing slash on source tells rsync to copy contents, not the directory
    itself, so the result is always consistent regardless of whether remote_dest
    already exists.
    """
    src = local.rstrip("/") + "/"
    return subprocess.run(
        ["rsync", "-avz", "--progress", src, f"{host}:{remote_dest}"]
    ).returncode


def rsync_pull(
    host: str,
    remote_src: str,
    local_dest: str,
    exclude: Optional[List[str]] = None,
) -> int:
    """Pull a remote path to local via rsync, with optional exclusion patterns."""
    cmd = ["rsync", "-avz", "--progress"]
    for pattern in exclude or []:
        cmd += ["--exclude", pattern]
    cmd += [f"{host}:{remote_src}", local_dest]
    return subprocess.run(cmd).returncode
