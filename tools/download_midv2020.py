#!/usr/bin/env python3
"""Download MIDV-2020 archives after access has been granted by the provider.

The official service requires accepting its licence and then supplies sFTP
credentials. This helper deliberately does not contain credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import shlex
import subprocess
import tarfile
import urllib.request
from pathlib import Path


def download_http(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request) as response:
        resumed = offset and response.status == 206
        mode = "ab" if resumed else "wb"
        if not resumed:
            offset = 0
        with partial.open(mode) as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    partial.replace(destination)


def download_sftp(remote: str, destination: Path, port: int, identity_file: Path | None) -> None:
    if ":" not in remote:
        raise ValueError("--sftp must look like user@host:/remote/path/archive.tar")
    target, remote_path = remote.split(":", 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["sftp", "-P", str(port)]
    if identity_file:
        command += ["-i", str(identity_file)]
    command += ["-b", "-", target]
    batch = f"get -p {shlex.quote(remote_path)} {shlex.quote(destination.name)}\n"
    subprocess.run(command, input=batch, text=True, cwd=destination.parent, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        tar.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="HTTP(S) archive URL, if a public mirror is available")
    source.add_argument("--sftp", help="user@host:/remote/archive.tar")
    parser.add_argument("--out", type=Path, required=True, help="local archive or dataset directory")
    parser.add_argument("--archive-name", default="midv2020.tar", help="name used with --url/--sftp")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--extract-dir", type=Path)
    args = parser.parse_args()

    if args.sftp:
        archive = args.out / args.archive_name
        download_sftp(args.sftp, archive, args.port, args.identity_file)
    else:
        archive = args.out / args.archive_name
        download_http(args.url, archive)
    if args.sha256 and sha256(archive).lower() != args.sha256.lower():
        raise SystemExit(f"SHA-256 mismatch for {archive}")
    if args.extract:
        destination = args.extract_dir or args.out
        destination.mkdir(parents=True, exist_ok=True)
        safe_extract(archive, destination)
    print(f"Downloaded {archive}")


if __name__ == "__main__":
    main()
