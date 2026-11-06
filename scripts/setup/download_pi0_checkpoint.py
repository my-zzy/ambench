# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Download the official pi0_base objects with resumable transfers and MD5 verification."""
import argparse
import base64
import fcntl
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--destination", type=Path, required=True)
parser.add_argument(
    "--manifest",
    type=Path,
    default=Path(__file__).resolve().parents[2] / "experiments/pressbutton/pi0_base_manifest.json",
)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--verify-only", action="store_true")
args = parser.parse_args()
metadata = json.loads(args.manifest.read_text())
assert not metadata.get("nextPageToken"), "Object listing is incomplete"
objects = metadata["items"]
destination = args.destination.resolve()
staging = destination.with_name(destination.name + ".partial")
setup_dir = destination.parent
if args.verify_only and not destination.is_dir():
    parser.error("Checkpoint is missing; verify-only never downloads")
setup_dir.mkdir(parents=True, exist_ok=True)
lock = (setup_dir / "checkpoint_download.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX)
if not destination.exists():
    staging.mkdir(parents=True, exist_ok=True)


def checksum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()


def prepare(item: dict) -> tuple:
    relative = Path(item["name"]).relative_to("checkpoints/pi0_base")
    assert ".." not in relative.parts
    path = staging / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(item["size"])
    if path.exists() and path.stat().st_size == expected_bytes and checksum(path) == item["md5Hash"]:
        return item, path, None, []
    partial = path.with_name(path.name + ".download")
    offset = partial.stat().st_size if partial.exists() else 0
    assert offset <= expected_bytes, relative
    url = (
        "https://storage.googleapis.com/download/storage/v1/b/openpi-assets/o/"
        + quote(item["name"], safe="")
        + "?alt=media&generation="
        + item["generation"]
    )
    segments = []
    for start in range(offset, expected_bytes, 8 * 1024 * 1024):
        end = min(start + 8 * 1024 * 1024, expected_bytes) - 1
        segment = staging / "_segments" / relative / f"{start}-{end}.part"
        segment.parent.mkdir(parents=True, exist_ok=True)
        segments.append((url, start, end, segment))
    return item, path, partial, segments


def download_segment(job: tuple) -> int:
    url, start, end, segment = job
    expected = end - start + 1
    if segment.exists() and segment.stat().st_size == expected:
        return expected
    temporary = segment.with_suffix(".partial")
    interrupted_piece = temporary.with_suffix(".piece")
    if interrupted_piece.exists():
        have = temporary.stat().st_size if temporary.exists() else 0
        assert have + interrupted_piece.stat().st_size <= expected
        with temporary.open("ab") as target, interrupted_piece.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        interrupted_piece.unlink()
    for attempt in range(20):
        have = temporary.stat().st_size if temporary.exists() else 0
        if have == expected:
            break
        assert have < expected, (temporary, have, expected)
        piece = temporary.with_suffix(".piece")
        result = subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "30",
                "--max-time",
                "1200",
                "--range",
                f"{start + have}-{end}",
                "--output",
                str(piece),
                url,
            ],
            capture_output=True,
            text=True,
        )
        if piece.exists():
            assert piece.stat().st_size <= expected - have, (piece, piece.stat().st_size)
            with temporary.open("ab") as target, piece.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            piece.unlink()
        if result.returncode:
            print(f"Range {start}-{end}, retry {attempt + 1}: {result.stderr.strip()}", flush=True)
            time.sleep(2)
    assert temporary.stat().st_size == expected, (segment, temporary.stat().st_size, expected)
    temporary.rename(segment)
    return expected


def assemble(prepared: tuple) -> dict:
    item, path, partial, segments = prepared
    if partial is not None:
        assembled = path.with_name(path.name + ".assembled")
        with assembled.open("wb") as target:
            sources = ([partial] if partial.exists() else []) + [job[3] for job in segments]
            for source in sources:
                with source.open("rb") as handle:
                    while chunk := handle.read(8 * 1024 * 1024):
                        target.write(chunk)
        assert assembled.stat().st_size == int(item["size"]), path
        assert checksum(assembled) == item["md5Hash"], f"Checksum mismatch: {path}"
        assembled.rename(path)
        for source in sources:
            source.unlink()
    relative = path.relative_to(staging)
    print(f"Verified {relative} ({int(item['size']):,} bytes)", flush=True)
    return {"path": str(relative), "bytes": int(item["size"]), "md5_base64": item["md5Hash"]}


if destination.exists():
    verified = []
    for item in objects:
        relative = Path(item["name"]).relative_to("checkpoints/pi0_base")
        path = destination / relative
        assert path.stat().st_size == int(item["size"]), path
        assert checksum(path) == item["md5Hash"], path
        verified.append({"path": str(relative), "bytes": int(item["size"]), "md5_base64": item["md5Hash"]})
    report = {"source": "gs://openpi-assets/checkpoints/pi0_base", "verified": True, "files": verified}
    (setup_dir / "checkpoint_download_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Existing checkpoint verified: {destination}", flush=True)
    raise SystemExit(0)
prepared = [prepare(item) for item in objects]
jobs = [job for item in prepared for job in item[3]]
started = time.monotonic()
received = 0
print(f"Downloading {len(jobs)} ranges with {args.workers} workers", flush=True)
with ThreadPoolExecutor(max_workers=args.workers) as executor:
    futures = [executor.submit(download_segment, job) for job in jobs]
    for count, future in enumerate(as_completed(futures), 1):
        received += future.result()
        if count % 16 == 0 or count == len(jobs):
            elapsed = time.monotonic() - started
            print(
                f"Ranges {count}/{len(jobs)}, {received / 1e9:.2f} GB, {received / elapsed / 1e6:.2f} MB/s", flush=True
            )
verified = [assemble(item) for item in prepared]
segment_root = staging / "_segments"
if segment_root.exists():
    for directory in sorted(segment_root.rglob("*"), reverse=True):
        if directory.is_dir():
            directory.rmdir()
    segment_root.rmdir()
staging.rename(destination)
report = {"source": "gs://openpi-assets/checkpoints/pi0_base", "verified": True, "files": verified}
(setup_dir / "checkpoint_download_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"Checkpoint ready: {destination}", flush=True)
