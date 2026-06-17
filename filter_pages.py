#!/usr/bin/env python3
"""Stage A - build one OCR batch per day for causal graph extraction."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE / "data"
OUT_DIR = WORKSPACE / "batches"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL = OUT_DIR / "batches.jsonl"

PAGE_MARKER = re.compile(r"^===== PAGE\s+(\d+)\s+=====\s*$", re.M)
DATE_STEM = re.compile(r"^(\d{4}-\d{2}-\d{2})$")

DEFAULT_LCCN = "sn83045462"
DEFAULT_EDITION = "ed-1"
DEFAULT_MAX_CHARS = 30000


def split_issue_pages(issue_text: str) -> list[tuple[int, str]]:
    matches = list(PAGE_MARKER.finditer(issue_text))
    if not matches:
        text = issue_text.strip()
        return [(1, text)] if text else []

    preamble = issue_text[:matches[0].start()].strip()
    pages: list[tuple[int, str]] = []
    for idx, match in enumerate(matches):
        seq = int(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(issue_text)
        page_text = issue_text[start:end].strip()
        if seq == 1 and preamble:
            page_text = f"{preamble}\n\n{page_text}".strip()
        if page_text:
            pages.append((seq, page_text))
    return pages


def iter_issue_files(data_dir: Path):
    for year_dir in sorted(data_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for issue_path in sorted(year_dir.glob("*.txt")):
            if DATE_STEM.match(issue_path.stem):
                yield issue_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                    help="maximum characters kept in the daily batch text")
    ap.add_argument("--limit", type=int, help="limit batches")
    ap.add_argument("--year", type=int, help="only process one year")
    args = ap.parse_args()

    kept = 0
    with OUT_JSONL.open("w", encoding="utf-8") as out:
        for issue_path in iter_issue_files(DATA_DIR):
            if args.year and issue_path.parent.name != str(args.year):
                continue
            date = issue_path.stem
            issue_text = issue_path.read_text(encoding="utf-8", errors="replace")
            rel_path = issue_path.relative_to(DATA_DIR).as_posix()
            pages = split_issue_pages(issue_text)
            batch_pages: list[dict] = [
                {
                    "seq": seq,
                    "mention_id": f"m:{date}:seq{seq:02d}",
                    "text": page_text,
                    "page_chars": len(page_text),
                }
                for seq, page_text in pages
            ]
            if not batch_pages:
                continue
            batch_text = "\n\n".join(
                f"===== PAGE {page['seq']} ({page['mention_id']}) =====\n{page['text']}"
                for page in batch_pages
            )
            rec = {
                "batch_id": f"batch:{date}:daily",
                "lccn": DEFAULT_LCCN,
                "issue_date": date,
                "edition": DEFAULT_EDITION,
                "issue_path": rel_path,
                "batch_index": 1,
                "start_seq": batch_pages[0]["seq"],
                "end_seq": batch_pages[-1]["seq"],
                "page_count": len(batch_pages),
                "page_count_total": len(pages),
                "pages": batch_pages,
                "batch_text": batch_text[: args.max_chars],
                "batch_chars": min(len(batch_text), args.max_chars),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            if args.limit and kept >= args.limit:
                print(f"[done] kept={kept} -> {OUT_JSONL}")
                return 0

    print(f"[done] kept={kept} -> {OUT_JSONL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())