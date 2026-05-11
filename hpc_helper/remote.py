from __future__ import annotations

import fnmatch
import os
import shlex
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import List, Optional, Tuple


def _load_hpcignore(directory: str) -> List[str]:
    """Return glob patterns from .hpcignore, trailing slashes stripped."""
    p = Path(directory) / ".hpcignore"
    if not p.exists():
        return []
    patterns = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip().rstrip("/")
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


def is_ignored(rel: str, patterns: List[str]) -> bool:
    """Return True if the relative path matches any .hpcignore pattern."""
    parts = rel.replace("\\", "/").split("/")
    for pattern in patterns:
        if any(fnmatch.fnmatch(part, pattern) for part in parts):
            return True
        if fnmatch.fnmatch(rel, pattern):
            return True
    return False


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


def _popen_pipe(write_cmd: List[str], read_cmd: List[str]):
    """Start write_cmd | read_cmd, return (writer, reader) Popen objects."""
    writer = subprocess.Popen(write_cmd, stdout=subprocess.PIPE)
    reader = subprocess.Popen(read_cmd, stdin=writer.stdout)
    writer.stdout.close()
    return writer, reader


def _wait_pipe(writer: subprocess.Popen, reader: subprocess.Popen) -> int:
    """Wait for both sides of a pipe, terminating both on interrupt."""
    try:
        reader.wait()
        writer.wait()
    except BaseException:
        writer.terminate()
        reader.terminate()
        writer.wait()
        reader.wait()
        raise
    return reader.returncode


def tar_push(
    local: str,
    host: str,
    remote_dest: str,
    files: Optional[List[str]] = None,
) -> int:
    """Push to remote_dest via system tar + SSH.

    files=None  → full push (everything minus .hpcignore).
    files=[...]  → incremental push: only the listed relative paths.

    PAX format ensures Unicode/Chinese filenames transfer correctly.
    """
    local_abs = str(Path(local).resolve())
    remote_cmd = (
        f"mkdir -p {shlex.quote(remote_dest)} && "
        f"LC_ALL=C.UTF-8 tar xzf - -C {shlex.quote(remote_dest)} 2>/dev/null"
    )

    if files is None:
        patterns = _load_hpcignore(local_abs)
        tar_cmd = ["tar", "-czf", "-", "--format=pax"]
        for p in patterns:
            tar_cmd += ["--exclude", p]
        tar_cmd += ["-C", local_abs, "."]
        writer, reader = _popen_pipe(tar_cmd, ["ssh", host, remote_cmd])
        return _wait_pipe(writer, reader)

    # Write file list to a temp file to avoid Windows command-line length limits
    # (100k paths easily exceeds the ~32k char limit if passed as arguments).
    # --files-from works now that we use -czf (with dash) instead of bare czf.
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    try:
        tmp.write("\n".join(files))
        tmp.close()
        tar_cmd = ["tar", "-czf", "-", "--format=pax", "-C", local_abs,
                   f"--files-from={tmp.name}"]
        writer, reader = _popen_pipe(tar_cmd, ["ssh", host, remote_cmd])
        return _wait_pipe(writer, reader)
    finally:
        os.unlink(tmp.name)


def tar_pull(
    host: str,
    remote_src: str,
    local_dest: str,
    exclude: Optional[List[str]] = None,
) -> int:
    """Pull a remote directory to local via system tar + SSH.

    Packs the directory *itself* on the remote so the result under local_dest
    mirrors the remote layout:
        remote: .../results/     →  local: local_dest/results/
    """
    remote_path = PurePosixPath(remote_src.rstrip("/"))
    remote_parent = shlex.quote(str(remote_path.parent))
    remote_name = shlex.quote(remote_path.name)
    excl = " ".join(f"--exclude={shlex.quote(p)}" for p in (exclude or []))
    remote_cmd = (
        f"LC_ALL=C.UTF-8 tar -czf - --format=pax {excl} -C {remote_parent} {remote_name}"
    )

    local_abs = str(Path(local_dest).resolve())
    Path(local_dest).mkdir(parents=True, exist_ok=True)

    writer, reader = _popen_pipe(
        ["ssh", host, remote_cmd],
        ["tar", "-xzf", "-", "-C", local_abs],
    )
    return _wait_pipe(writer, reader)
