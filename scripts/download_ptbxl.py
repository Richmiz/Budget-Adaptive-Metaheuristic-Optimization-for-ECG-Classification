"""Download PTB-XL v1.0.3 (open access, CC BY 4.0).

Default: the metadata plus the **100 Hz records only** (the study's input, protocol §3.1) from
PhysioNet's open-data mirror on AWS S3. Files are fetched in parallel, verified against
PhysioNet's SHA256SUMS.txt, and skipped when already present and valid, so the script can be
re-run to resume.

    python scripts/download_ptbxl.py                 # 100 Hz subset (~0.5 GB) from the S3 mirror
    python scripts/download_ptbxl.py --with-500hz    # also the 500 Hz records
    python scripts/download_ptbxl.py --zip           # full archive from physionet.org (1.84 GB, slow)
"""

import _paths  # noqa: F401

import argparse
import csv
import hashlib
import shutil
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bamo.utils import sha256_file, write_json

S3_BASE = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
ZIP_URL = ("https://physionet.org/static/published-projects/ptb-xl/"
           "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip")
META_FILES = ["SHA256SUMS.txt", "ptbxl_database.csv", "scp_statements.csv", "LICENSE.txt", "RECORDS",
              "example_physionet.py", "ptbxl_v102_changelog.txt", "ptbxl_v103_changelog.txt"]
TARGET_NAME = "ptb-xl-1.0.3"


def fetch(url: str, dest: Path, expected_sha: str | None, retries: int = 5) -> int:
    """Download ``url`` to ``dest`` (atomic rename), verify SHA-256, return bytes written."""
    for attempt in range(retries):
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
                while chunk := r.read(1 << 16):
                    h.update(chunk)
                    f.write(chunk)
            if expected_sha and h.hexdigest() != expected_sha:
                raise ValueError(f"checksum mismatch for {dest.name}")
            tmp.replace(dest)
            return dest.stat().st_size
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return 0


def read_checksums(path: Path) -> dict[str, str]:
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.strip().partition(" ")
        if name:
            sums[name.strip()] = digest
    return sums


def download_subset(dest: Path, with_500hz: bool, workers: int) -> dict:
    root = dest / TARGET_NAME
    root.mkdir(parents=True, exist_ok=True)
    fetch(S3_BASE + "SHA256SUMS.txt", root / "SHA256SUMS.txt", None)
    sums = read_checksums(root / "SHA256SUMS.txt")
    for name in META_FILES[1:]:
        fetch(S3_BASE + name, root / name, sums.get(name))

    with (root / "ptbxl_database.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    columns = ["filename_lr"] + (["filename_hr"] if with_500hz else [])
    files = [f"{row[c]}{ext}" for row in rows for c in columns for ext in (".hea", ".dat")]
    missing_sums = [name for name in files if name not in sums]
    if missing_sums:
        raise SystemExit(f"{len(missing_sums)} record files have no checksum, e.g. {missing_sums[:3]}")

    todo = []
    for name in files:
        path = root / name
        if path.exists() and sha256_file(path) == sums[name]:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        todo.append(name)
    print(f"{len(files)} record files listed, {len(files) - len(todo)} already verified, {len(todo)} to download")

    lock = threading.Lock()
    done, nbytes, failures, t0 = 0, 0, [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, S3_BASE + name, root / name, sums[name]): name for name in todo}
        for fut in as_completed(futures):
            try:
                size = fut.result()
                with lock:
                    nbytes += size
            except Exception as err:
                failures.append({"file": futures[fut], "error": repr(err)})
            done += 1
            if done % 2000 == 0 or done == len(todo):
                rate = nbytes / max(time.time() - t0, 1e-9) / 2**20
                print(f"  {done}/{len(todo)} files, {nbytes / 2**20:.0f} MB, {rate:.2f} MB/s, {len(failures)} failures",
                      flush=True)
    return {"source": S3_BASE, "subset": "100Hz+500Hz" if with_500hz else "100Hz", "n_record_files": len(files),
            "downloaded_files": len(todo) - len(failures), "downloaded_bytes": nbytes, "failures": failures,
            "sha256sums_sha256": sha256_file(root / "SHA256SUMS.txt"), "extracted_to": str(root)}


def download_zip(dest: Path) -> dict:
    archive = dest / Path(ZIP_URL).name
    if not archive.exists():
        print(f"downloading {ZIP_URL}")
        with urllib.request.urlopen(ZIP_URL) as r, archive.open("wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
    digest = sha256_file(archive)
    with zipfile.ZipFile(archive) as z:
        top = z.namelist()[0].split("/")[0]
        z.extractall(dest)
    target = dest / TARGET_NAME
    if not target.exists():
        (dest / top).rename(target)
    return {"source": ZIP_URL, "subset": "full archive", "archive_sha256": digest, "extracted_to": str(target)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/raw")
    ap.add_argument("--zip", action="store_true", help="full archive from physionet.org instead of the S3 mirror")
    ap.add_argument("--with-500hz", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    info = download_zip(dest) if args.zip else download_subset(dest, args.with_500hz, args.workers)
    info["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(dest / "ptbxl_download.json", info)
    if info.get("failures"):
        raise SystemExit(f"{len(info['failures'])} files failed; re-run to resume")
    print(f"done: {info['extracted_to']}")


if __name__ == "__main__":
    main()
