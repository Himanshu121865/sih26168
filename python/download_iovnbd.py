#!/usr/bin/env python3
"""
download_iovnbd.py — Step 1.2-1.4
Fetches IO-VNBD Synchronised + Unsynchronised zips via media.githubusercontent.com
Idempotent: skips if zip already exists and size matches HEAD.

Usage:
  python python/download_iovnbd.py --subset Sync          # 203 MB only (default, enough for Steps 2-5)
  python python/download_iovnbd.py --subset all           # both zips (203+214 MB)
  python python/download_iovnbd.py --out data/iovnbd      # custom out dir
  python python/download_iovnbd.py --unzip                # also unzip (default)
"""
import argparse, os, sys, hashlib
from pathlib import Path
import urllib.request

SYNC_URL = "https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/Synchronised%20V%20abd%20S%20datasets.zip"
UNSYNC_URL = "https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/Unsynchronised%20V%20and%20S%20Dataset.zip"
SYNC_SIZE = 203606286  # from HEAD
UNSYNC_SIZE = 214330231

def download(url: str, dest: Path, expected_size: int):
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"[skip] {dest.name} already {expected_size} bytes")
        return dest
    print(f"[dl] {url} -> {dest} ({expected_size/1e6:.1f} MB)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # stream with progress
    def report(block_num, block_size, total):
        if total > 0 and block_num % 100 == 0:
            pct = block_num * block_size / total * 100
            print(f"  {pct:.1f}%", end="\r")
    urllib.request.urlretrieve(url, dest, reporthook=report)
    print(f"\n[done] {dest} {dest.stat().st_size} bytes")
    if dest.stat().st_size != expected_size:
        print(f"[warn] size mismatch: got {dest.stat().st_size}, expected {expected_size}", file=sys.stderr)
    return dest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["Sync","all","Unsync"], default="Sync", help="which zips")
    ap.add_argument("--out", default="data/iovnbd", help="output dir for zips")
    ap.add_argument("--unzip", action="store_true", default=True, help="unzip after download")
    ap.add_argument("--no-unzip", dest="unzip", action="store_false")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.subset in ("Sync","all"):
        f = out / "Synchronised V abd S datasets.zip"
        download(SYNC_URL, f, SYNC_SIZE)
        if args.unzip:
            import zipfile
            print(f"[unzip] {f} -> {out}/")
            with zipfile.ZipFile(f) as z:
                z.extractall(out)
            print("[unzip] done", list((out / "Synchronised V abd S datasets").iterdir())[:3])
    if args.subset in ("Unsync","all"):
        f = out / "Unsynchronised V and S Dataset.zip"
        download(UNSYNC_URL, f, UNSYNC_SIZE)
        if args.unzip:
            import zipfile
            print(f"[unzip] {f} -> {out}/")
            with zipfile.ZipFile(f) as z:
                z.extractall(out)
    print("[ok] download_iovnbd done")

if __name__ == "__main__":
    main()
