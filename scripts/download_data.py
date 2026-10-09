#!/usr/bin/env python3
"""Safely download the UCI archive and extract only its expected CSV."""
from datetime import date
from hashlib import sha256
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile
import sys

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
CSV_NAME = "online_shoppers_intention.csv"
RAW = ROOT / "data" / "raw"
TARGET = RAW / CSV_NAME
README = ROOT / "data" / "README.md"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        print(f"Refusing to overwrite existing file: {TARGET}")
        return 0
    archive = RAW / ".online-shopper-download.zip"
    try:
        request = Request(URL, headers={"User-Agent": "online-shopper-analysis/1.0"})
        with urlopen(request, timeout=30) as response, archive.open("wb") as output:
            output.write(response.read())
        with ZipFile(archive) as zip_file:
            members = [m for m in zip_file.infolist() if Path(m.filename).name == CSV_NAME
                       and not m.is_dir()]
            if len(members) != 1:
                raise ValueError(f"Expected exactly one {CSV_NAME}, found {len(members)}")
            member = members[0]
            if Path(member.filename).name != CSV_NAME:
                raise ValueError("Unexpected archive member")
            with zip_file.open(member) as source, TARGET.open("xb") as destination:
                destination.write(source.read())
    except (HTTPError, URLError, TimeoutError, OSError, BadZipFile, ValueError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        print(f"Manual fallback: download the UCI archive and place {CSV_NAME} in {RAW}.",
              file=sys.stderr)
        return 1
    finally:
        archive.unlink(missing_ok=True)
    digest = sha256(TARGET.read_bytes()).hexdigest()
    with README.open("a", encoding="utf-8") as output:
        output.write(f"\n## Retrieval record\n- Date: {date.today().isoformat()}\n"
                     f"- Source: {URL}\n- SHA-256: `{digest}`\n")
    print(f"Downloaded {TARGET} (SHA-256: {digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
