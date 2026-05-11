from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
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


def tar_push(local: str, host: str, remote_dest: str) -> int:
    """Push local directory contents into remote_dest via tar-over-SSH.

    Uses tar + SSH instead of rsync so it works on Windows (OpenSSH + the
    built-in bsdtar shipped with Windows 10+), macOS, and Linux alike.

    `tar -C localdir .` packs the *contents* of the directory, so the layout
    on the remote is always:
        remote_dest/file.py   (not remote_dest/localdir/file.py)
    regardless of whether remote_dest already exists.
    """
    local_abs = str(Path(local).resolve())
    remote_cmd = (
        f"mkdir -p {shlex.quote(remote_dest)} && "
        f"tar xzf - -C {shlex.quote(remote_dest)}"
    )
    tar = subprocess.Popen(
        ["tar", "czf", "-", "-C", local_abs, "."],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    ssh = subprocess.Popen(["ssh", host, remote_cmd], stdin=tar.stdout)
    tar.stdout.close()  # allow tar to receive SIGPIPE if ssh exits early
    ssh.wait()
    tar.wait()
    return ssh.returncode


def tar_pull(
    host: str,
    remote_src: str,
    local_dest: str,
    exclude: Optional[List[str]] = None,
) -> int:
    """Pull remote directory contents to local via tar-over-SSH.

    Exclusion patterns (e.g. 'slurm-*.out') are applied on the remote side
    before the archive is created, so they never consume bandwidth.
    """
    excl = " ".join(f"--exclude={shlex.quote(p)}" for p in (exclude or []))
    remote_cmd = f"tar czf - {excl} -C {shlex.quote(remote_src)} ."

    local_abs = str(Path(local_dest).resolve())
    Path(local_dest).mkdir(parents=True, exist_ok=True)

    ssh = subprocess.Popen(["ssh", host, remote_cmd], stdout=subprocess.PIPE)
    tar = subprocess.Popen(["tar", "xzf", "-", "-C", local_abs], stdin=ssh.stdout)
    ssh.stdout.close()
    tar.wait()
    ssh.wait()
    return tar.returncode
