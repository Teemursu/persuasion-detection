"""Download the Multilingual Persuasion Dataset from Zenodo.

Record: https://zenodo.org/records/6341173  (DOI 10.5281/zenodo.6341173, CC BY 4.0)

The file list and MD5 checksums are read from the Zenodo REST API, so the
script keeps working if the record gets a new file. Files are saved next to
this script (``data/``), which is git-ignored.

Usage:
    python data/download.py            # download missing files, verify MD5
    python data/download.py --force    # re-download everything
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

RECORD_ID = "6341173"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
DATA_DIR = Path(__file__).resolve().parent


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_record() -> dict:
    req = urllib.request.Request(API_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = ap.parse_args()

    record = fetch_record()
    meta = record["metadata"]
    print(f"Zenodo record {RECORD_ID}: {meta['title']}")
    print(f"  DOI: {record.get('doi', meta.get('doi'))}   licence: {meta.get('license', {}).get('id')}")

    ok = True
    for f in record["files"]:
        name, size = f["key"], f["size"]
        algo, _, expected = f["checksum"].partition(":")
        dest = DATA_DIR / name
        if dest.exists() and not args.force and md5sum(dest) == expected:
            print(f"- {name}: already present, checksum OK")
            continue
        print(f"- {name} ({size / 1e6:.1f} MB): downloading")
        download(f["links"]["self"], dest)
        got = md5sum(dest) if algo == "md5" else None
        if got is not None and got != expected:
            print(f"  checksum mismatch: expected {expected}, got {got}", file=sys.stderr)
            ok = False
        else:
            print("  checksum OK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
