#!/usr/bin/env python3
"""Fetch and verify the inputs listed in data_sources.yaml.

Run from reproducibility/::

    python -m fetch_data                 # fetch everything missing, verify all
    python -m fetch_data --arm clustering
    python -m fetch_data --list          # what is needed, what is present
    python -m fetch_data --check         # verify what exists, download nothing

A file whose sha256 already matches is left alone. A file with the wrong digest is reported and kept. Pass --force to
replace it.

Exit codes: 0 when every requested input is present and verified, 1 when an input is missing or failed to verify,
2 when a file exists but its digest does not match.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import tarfile
import urllib.request

import yaml

HERE = pathlib.Path(__file__).resolve().parent
MANIFEST = HERE / "data_sources.yaml"
CHUNK = 1 << 20
USER_AGENT = "Mozilla/5.0 (compatible; lntest-reproducibility/1.0)"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def load_sources(arm=None):
    entries = yaml.safe_load(MANIFEST.read_text())["sources"]
    if arm:
        entries = [e for e in entries if arm in e.get("arms", [])]
        if not entries:
            sys.exit(f"no entry in {MANIFEST.name} lists arm {arm!r}")
    return entries


def status(entry):
    """(state, detail) for one entry, without touching the network."""
    dest = HERE / entry["dest"]
    if not dest.exists():
        return "MISSING", ""
    size = dest.stat().st_size
    want_bytes = entry.get("bytes")
    want = entry.get("sha256")

    if want is None:
        # The multi-GB spatial archives are pinned by size only, which catches
        # a truncated download.
        if want_bytes is None:
            return "UNPINNED", f"present, {size} bytes, nothing to check against"
        if size == want_bytes:
            return "SIZE-OK", f"{size} bytes, size only, no sha256"
        return "MISMATCH", f"have {size} bytes, want {want_bytes}"

    if want_bytes is not None and size != want_bytes:
        return "MISMATCH", f"have {size} bytes, want {want_bytes}"
    got = sha256(dest)
    if got == want:
        return "OK", ""
    return "MISMATCH", f"have {got[:16]}..., want {want[:16]}..."


def download(entry, dest: pathlib.Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {entry['url']}")
    # Written to .part until complete, so an interrupted download is not taken
    # for a present file.
    tmp = dest.with_suffix(dest.suffix + ".part")
    # 10x's CDN refuses urllib's default User-Agent.
    req = urllib.request.Request(entry["url"], headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as r, tmp.open("wb") as fh:
        while block := r.read(CHUNK):
            fh.write(block)
    tmp.replace(dest)


def extract(entry, dest: pathlib.Path):
    target = HERE / entry["extract_to"]
    target.mkdir(parents=True, exist_ok=True)
    print(f"  extracting into {entry['extract_to']}/")
    with tarfile.open(dest) as tf:
        tf.extractall(target, filter="data")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", help="only the inputs of one experiment directory")
    ap.add_argument("--list", action="store_true", help="report status, change nothing")
    ap.add_argument("--check", action="store_true", help="verify what exists, download nothing")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    entries = load_sources(args.arm)
    rc = 0

    for entry in entries:
        dest = HERE / entry["dest"]
        state, detail = status(entry)

        if args.list or args.check:
            print(f"{state:9s} {entry['name']:14s} {entry['dest']}"
                  + (f"  ({detail})" if detail else ""))
            if state == "MISMATCH":
                rc = max(rc, 2)
            elif state == "MISSING":
                rc = max(rc, 1)
            continue

        if state in ("OK", "SIZE-OK") and not args.force:
            print(f"{state:9s} {entry['name']:14s} {entry['dest']}"
                  + (f"  ({detail})" if detail else ""))
            continue
        if state == "MISMATCH" and not args.force:
            print(f"MISMATCH  {entry['name']:14s} {entry['dest']}  ({detail})")
            print("          refusing to overwrite; pass --force to replace it")
            rc = max(rc, 2)
            continue
        if state == "UNPINNED" and not args.force:
            print(f"UNPINNED  {entry['name']:14s} {entry['dest']}  ({detail})")
            continue

        print(f"FETCH     {entry['name']:14s} {entry['dest']}")
        try:
            download(entry, dest)
        except Exception as exc:  # noqa: BLE001 - report and continue to the next input
            print(f"          FAILED: {type(exc).__name__}: {exc}")
            rc = max(rc, 1)
            continue

        want = entry.get("sha256")
        if want is None:
            state, detail = status(entry)
            print(f"          fetched, {state}"
                  + (f" ({detail})" if detail else ""))
            if state == "MISMATCH":
                rc = max(rc, 2)
                continue
        else:
            got = sha256(dest)
            if got != want:
                print(f"          FAILED verification: got {got}, want {want}")
                rc = max(rc, 2)
                continue
            print("          verified")

        if "extract_to" in entry:
            extract(entry, dest)

    return rc


if __name__ == "__main__":
    sys.exit(main())
