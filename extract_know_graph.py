#!/usr/bin/env python3
"""
Stage B — batch-level causal graph extraction with a text-only 7B model.

입력:  batches/batches.jsonl (Stage A 출력)
    또는 data/<year>/<date>.txt 를 contiguous page batch 로 직접 분할한 fallback
출력:  batch_graphs/<batch_id>.json   (배치당 1 JSON)
       batch_graphs/_index.jsonl      (성공/실패 인덱스)
       batch_graphs/_errors.jsonl     (파싱 실패 배치)

특징:
    1. checkpoint/resume — 이미 처리한 페이지는 건너뜀
    2. JSON 검증·재시도 — 잘못된 JSON이면 temperature 살짝 올려 재시도 (최대 2회)
    3. 스키마 검증 — node type, edge rel, topic vocab 검사 (느슨하게: 경고만)
    4. OOM 방어 — 페이지가 너무 길면 청크 분할
    5. 진행 상황 로그

사용법:
    # 0) 환경 설치 (한 번만)
    pip install transformers>=4.45 accelerate sentencepiece einops timm \\
                torch torchvision flash-attn

    # 1) 50페이지 스모크 테스트 (1917년 4월 중심)
    python3 extract_ekg.py --smoke

    # 2) 전체 실행
    python3 extract_ekg.py

    # 3) 특정 날짜 범위만
    python3 extract_ekg.py --start 1917-04-01 --end 1917-08-31

    # 4) 멈춘 곳부터 재개 (기본 동작 — 자동)
    python3 extract_ekg.py

GPU 권장: RTX 4090 (24GB).  메모리 사용 ≈ 17–19GB (BF16).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Iterable

try:
    from json_repair import repair_json  # type: ignore
except Exception:  # pragma: no cover
    repair_json = None

# 프로젝트 루트 추가
WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

from prompts.extraction_prompt import (  # noqa: E402
    build_messages,
    ALLOWED_TOPICS, ALLOWED_ROLES, ALLOWED_NODE_TYPES, ALLOWED_EDGE_RELS,
)

# --------------------------------------------------------------------------- #
# 경로
# --------------------------------------------------------------------------- #
SOURCE_DIR = WORKSPACE / "data"
FILTERED_JSONL = WORKSPACE / "filtered" / "filtered_pages.jsonl"

OUT_DIR = WORKSPACE / "batch_graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_JSONL = OUT_DIR / "_index.jsonl"
ERROR_JSONL = OUT_DIR / "_errors.jsonl"

# --------------------------------------------------------------------------- #
# 모델 로딩
# --------------------------------------------------------------------------- #
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_INPUT_CHARS = 12000          # 페이지가 너무 길면 자름 (≈3K tokens)
MAX_NEW_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.1        # 추출은 결정적이어야 함
RETRY_TEMPERATURES = [0.1, 0.4]  # 1차/2차 재시도 시
DEFAULT_LCCN = "sn83045462"
DEFAULT_EDITION = "ed-1"
BATCH_JSONL = WORKSPACE / "batches" / "batches.jsonl"
PAGE_MARKER = re.compile(r"^===== PAGE\s+(\d+)\s+=====\s*$", re.M)
DATE_STEM = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
DEFAULT_MAX_PAGES = 64
DEFAULT_MAX_CHARS = 30000


@dataclass
class Generator:
    """모델 + 토크나이저 wrapper."""
    tokenizer: object
    model: object
    device: str

    def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        """messages → 모델 응답 텍스트."""
        import torch  # type: ignore
        # InternVL3는 transformers의 chat_template을 지원
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # 새 토큰만 디코드
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


def load_model(dtype: str = "bf16") -> Generator:
    """Text-only 7B model loading (BF16, GPU)."""
    import torch  # type: ignore
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

    print(f"[model] loading {MODEL_ID} (dtype={dtype})...")
    t0 = time.time()
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    except Exception as exc:
        print(f"[model] direct load failed, retrying with device_map='auto': {exc}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    # VRAM 사용량 측정
    if torch.cuda.is_available():
        gb = torch.cuda.memory_allocated() / 1e9
        print(f"[model] loaded in {time.time()-t0:.1f}s, VRAM used: {gb:.1f} GB")
        device = "cuda"
    else:
        print(f"[model] loaded in {time.time()-t0:.1f}s (CPU mode — VERY SLOW)")
        device = "cpu"

    return Generator(tokenizer=tokenizer, model=model, device=device)


# --------------------------------------------------------------------------- #
# JSON 파싱 — 모델 출력에서 첫 JSON 객체만 안전하게 뽑기
# --------------------------------------------------------------------------- #
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """모델 출력에서 JSON 객체 추출. 실패하면 None."""
    # 1) ```json ... ``` 펜스 우선
    m = _JSON_FENCE.search(text)
    if m:
        candidate = m.group(1).strip()
    else:
        # 2) 첫 '{' 부터 마지막 '}' 까지
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start:end + 1]
    # 3) 파싱 시도
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # trailing comma 등 사소한 오류 한 번 더 시도
        try:
            cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            if repair_json is not None:
                try:
                    repaired = repair_json(candidate)
                    return json.loads(repaired)
                except Exception:
                    return None
            return None


# --------------------------------------------------------------------------- #
# 스키마 검증 — 느슨하게 (경고만, drop 안 함)
# --------------------------------------------------------------------------- #
def validate_extraction(data: dict, expected_mention_id: str) -> list[str]:
    """스키마 위반 사항을 경고 리스트로 반환. 빈 리스트면 OK."""
    warns: list[str] = []
    if not isinstance(data, dict):
        return ["root is not dict"]
    if "batch" not in data or "nodes" not in data or "edges" not in data:
        warns.append("missing top-level keys")
        return warns
    if data["batch"].get("id") != expected_mention_id:
        warns.append(f"batch id mismatch: got {data['batch'].get('id')}")

    seen_ids: set[str] = set()
    for n in data.get("nodes", []):
        if not isinstance(n, dict):
            warns.append("node not dict")
            continue
        nt = n.get("type")
        if nt not in ALLOWED_NODE_TYPES:
            warns.append(f"unknown node type: {nt}")
        if nt == "Topic":
            tid = (n.get("id") or "").removeprefix("topic:")
            if tid and tid not in ALLOWED_TOPICS:
                warns.append(f"topic outside vocab: {tid}")
        if nt == "Role":
            rid = (n.get("id") or "").removeprefix("role:")
            if rid and rid not in ALLOWED_ROLES:
                warns.append(f"role outside vocab: {rid}")
        if n.get("id"):
            seen_ids.add(n["id"])

    for e in data.get("edges", []):
        if not isinstance(e, dict):
            warns.append("edge not dict")
            continue
        if e.get("rel") not in ALLOWED_EDGE_RELS:
            warns.append(f"unknown edge rel: {e.get('rel')}")
        # 엔드포인트가 노드에 존재하는지 (lenient — Stage C에서 fixup 가능)
        if e.get("from") and e["from"] not in seen_ids:
            warns.append(f"edge 'from' not in nodes: {e['from']}")
        if e.get("to") and e["to"] not in seen_ids:
            warns.append(f"edge 'to' not in nodes: {e['to']}")

    return warns


# --------------------------------------------------------------------------- #
# 페이지 입력 준비
# --------------------------------------------------------------------------- #
def detect_genre(text: str) -> str:
    """간이 장르 감지 — prompt hint용."""
    t = text.lower()[:2000]
    if "editorial" in t or "the writer" in t:
        return "editorial"
    if "to the editor" in t or "letter to the editor" in t:
        return "letter"
    if "advertisement" in t or "for sale" in t or "want ad" in t:
        return "advertisement"
    if "cartoon" in t:
        return "cartoon"
    return "news"


def make_mention_id(date: str, seq: int) -> str:
    return f"m:{date}:seq{seq:02d}"


def page_output_path(date: str, seq: int) -> Path:
    return OUT_DIR / date / f"seq{seq:02d}.json"


def split_issue_pages(issue_text: str) -> list[tuple[int, str]]:
    """한 이슈 텍스트를 PAGE 마커 기준으로 분리한다."""
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


def batch_pages(pages: list[tuple[int, str]],
                max_pages: int = DEFAULT_MAX_PAGES,
                max_chars: int = DEFAULT_MAX_CHARS) -> list[list[dict]]:
    """연속 페이지를 배치로 묶는다."""
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_chars = 0
    for seq, page_text in pages:
        page_rec = {
            "seq": seq,
            "mention_id": None,
            "text": page_text,
            "page_chars": len(page_text),
        }
        prospective = current_chars + len(page_text)
        if current and (len(current) >= max_pages or prospective > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(page_rec)
        current_chars += len(page_text)
    if current:
        batches.append(current)
    return batches


def score_page(text: str) -> int:
    """가벼운 키워드 점수. Stage A 필터와 smoke test의 기준으로 사용."""
    t = text.lower()
    weights = [
        ("prohibition", 5),
        ("temperance", 5),
        ("volstead", 5),
        ("anti-saloon", 5),
        ("18th amendment", 5),
        ("eighteenth amendment", 5),
        ("wayne wheeler", 4),
        ("saloon", 3),
        ("wet", 2),
        ("dry", 2),
        ("liquor", 2),
        ("beer", 2),
        ("brewing", 2),
        ("alcohol", 2),
        ("war", 1),
        ("german", 1),
        ("food conservation", 1),
        ("hoover", 1),
        ("women suffrage", 1),
        ("nativism", 1),
        ("immigration", 1),
        ("public health", 1),
        ("moral", 1),
    ]
    score = 0
    for needle, weight in weights:
        if needle in t:
            score += weight
    return score


def _resolve_filtered_text(rec: dict) -> str:
    page_text = rec.get("page_text") or rec.get("text")
    if page_text:
        return page_text
    path_value = rec.get("path")
    if not path_value:
        return ""
    page_path = Path(path_value)
    if not page_path.is_absolute():
        page_path = SOURCE_DIR / page_path
    if not page_path.exists():
        return ""
    return page_path.read_text(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# 처리 루프
# --------------------------------------------------------------------------- #
@dataclass
class BatchJob:
    batch_id: str
    lccn: str
    issue_date: str
    edition: str
    issue_path: str
    batch_index: int
    start_seq: int
    end_seq: int
    pages: list[dict]
    text: str


def build_batch_record(issue_date: str, issue_path: str, batch_index: int,
                       pages: list[tuple[int, str]], max_chars: int = DEFAULT_MAX_CHARS) -> dict:
    batch_pages = []
    for seq, page_text in pages:
        mention_id = f"m:{issue_date}:seq{seq:02d}"
        batch_pages.append({
            "seq": seq,
            "mention_id": mention_id,
            "text": page_text[:max_chars],
            "page_chars": len(page_text),
        })
    start_seq = batch_pages[0]["seq"]
    end_seq = batch_pages[-1]["seq"]
    batch_id = f"batch:{issue_date}:issue{batch_index:02d}:p{start_seq:02d}-{end_seq:02d}"
    batch_text = "\n\n".join(
        f"===== PAGE {page['seq']} ({page['mention_id']}) =====\n{page['text']}"
        for page in batch_pages
    )
    return {
        "batch_id": batch_id,
        "lccn": DEFAULT_LCCN,
        "issue_date": issue_date,
        "edition": DEFAULT_EDITION,
        "issue_path": issue_path,
        "batch_index": batch_index,
        "start_seq": start_seq,
        "end_seq": end_seq,
        "page_count": len(batch_pages),
        "pages": batch_pages,
        "batch_text": batch_text,
        "batch_chars": len(batch_text),
    }


def iter_jobs(start: str | None, end: str | None,
              limit: int | None = None) -> Iterable[BatchJob]:
    """batches.jsonl 에서 작업 생성하거나 data/에서 배치를 직접 구성한다."""
    n_yielded = 0
    def emit(rec: dict) -> BatchJob | None:
        d = rec["issue_date"]
        if start and d < start:
            return None
        if end and d > end:
            return None
        if not rec.get("batch_text"):
            return None
        batch_text = rec["batch_text"]
        if len(batch_text) > MAX_INPUT_CHARS:
            batch_text = batch_text[:MAX_INPUT_CHARS] + "\n[...TRUNCATED...]"
        return BatchJob(
            batch_id=rec["batch_id"],
            lccn=rec.get("lccn", DEFAULT_LCCN),
            issue_date=d,
            edition=rec.get("edition", DEFAULT_EDITION),
            issue_path=rec.get("issue_path", ""),
            batch_index=int(rec.get("batch_index", 1)),
            start_seq=int(rec.get("start_seq", 1)),
            end_seq=int(rec.get("end_seq", 1)),
            pages=rec.get("pages", []),
            text=batch_text,
        )

    if BATCH_JSONL.exists():
        with BATCH_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                job = emit(rec)
                if job is None:
                    continue
                yield job
                n_yielded += 1
                if limit and n_yielded >= limit:
                    return
        return

    if not SOURCE_DIR.exists():
        print(f"[error] {SOURCE_DIR} not found — no input corpus available.",
              file=sys.stderr)
        sys.exit(1)

    for year_dir in sorted(SOURCE_DIR.iterdir()):
        if not year_dir.is_dir():
            continue
        for issue_path in sorted(year_dir.glob("*.txt")):
            m = DATE_STEM.match(issue_path.stem)
            if not m:
                continue
            issue_date = m.group(1)
            if start and issue_date < start:
                continue
            if end and issue_date > end:
                continue
            issue_text = issue_path.read_text(encoding="utf-8", errors="replace")
            pages = split_issue_pages(issue_text)
            rel_path = issue_path.relative_to(SOURCE_DIR).as_posix()
            if not pages:
                continue
            for batch_index, start in enumerate(range(0, len(pages), DEFAULT_MAX_PAGES), 1):
                chunk = pages[start:start + DEFAULT_MAX_PAGES]
                rec = build_batch_record(issue_date, rel_path, batch_index, chunk)
                job = emit(rec)
                if job is None:
                    continue
                yield job
                n_yielded += 1
                if limit and n_yielded >= limit:
                    return


def extract_one(gen: Generator, job: BatchJob) -> tuple[dict | None, list[str], str]:
    """한 배치 추출. (json_or_none, warnings, raw_response)."""
    raw = ""
    for attempt, temp in enumerate(RETRY_TEMPERATURES, 1):
        messages = build_messages(
            batch_id=job.batch_id, lccn=job.lccn, issue_date=job.issue_date,
            edition=job.edition, pages=job.pages, batch_text=job.text,
        )
        try:
            raw = gen.chat(messages, temperature=temp)
        except Exception as e:
            return None, [f"chat error: {e}"], ""
        data = extract_json(raw)
        if data is not None:
            warns = validate_extraction(data, job.batch_id)
            return data, warns, raw
        # JSON 실패 — 재시도 (마지막 시도면 종료)
        if attempt == len(RETRY_TEMPERATURES):
            return None, ["JSON parse failed after retries"], raw
    return None, ["unreachable"], raw


def process(start: str | None, end: str | None, limit: int | None,
            smoke: bool, dry_run: bool) -> int:
    """전체 처리 루프."""
    # 1) 이미 처리한 페이지 (resume)
    done: set[str] = set()
    if INDEX_JSONL.exists():
        with INDEX_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") == "ok":
                        done.add(r["batch_id"])
                except json.JSONDecodeError:
                    continue
        print(f"[resume] {len(done)} pages already processed — skipping.")

    # 2) 모델 로딩 (dry-run이면 생략)
    gen: Generator | None = None
    if not dry_run:
        gen = load_model()

    # 3) 잡 생성
    if smoke:
        start = start or "1917-04-01"
        end = end or "1917-04-30"
        limit = limit or 20
    jobs = list(iter_jobs(start, end, limit))
    print(f"[plan] {len(jobs)} pages to process "
          f"(range: {start or '·'} → {end or '·'}, limit={limit})")

    if dry_run:
        # 첫 잡의 messages 출력 (디버깅용)
        if jobs:
            j = jobs[0]
            msgs = build_messages(j.batch_id, j.lccn, j.issue_date,
                                  j.edition, j.pages, j.text)
            sys_len = len(msgs[0]["content"])
            user_len = len(msgs[-1]["content"])
            print(f"[dry-run] first job: {j.batch_id} pages={len(j.pages)}")
            print(f"[dry-run] system: {sys_len:,} chars,  "
                  f"final user: {user_len:,} chars,  "
                  f"total turns: {len(msgs)}")
        return 0

    # 4) 처리
    n_ok = n_err = n_skip = 0
    t_start = time.time()
    assert gen is not None
    with INDEX_JSONL.open("a", encoding="utf-8") as idx_out, \
            ERROR_JSONL.open("a", encoding="utf-8") as err_out:
        for i, job in enumerate(jobs, 1):
            key = job.batch_id
            if key in done:
                n_skip += 1
                continue
            t0 = time.time()
            try:
                data, warns, raw = extract_one(gen, job)
            except Exception as e:
                tb = traceback.format_exc(limit=3)
                err_out.write(json.dumps({
                    "batch_id": job.batch_id,
                    "error": str(e), "traceback": tb,
                }, ensure_ascii=False) + "\n")
                err_out.flush()
                n_err += 1
                continue

            dt = time.time() - t0
            if data is None:
                err_out.write(json.dumps({
                    "batch_id": job.batch_id,
                    "error": warns[0] if warns else "unknown",
                    "raw_head": raw[:500],
                }, ensure_ascii=False) + "\n")
                err_out.flush()
                idx_out.write(json.dumps({
                    "batch_id": job.batch_id,
                    "status": "fail", "warns": warns, "dt": round(dt, 2),
                }, ensure_ascii=False) + "\n")
                idx_out.flush()
                n_err += 1
                print(f"  [{i:>4}/{len(jobs)}] {job.batch_id}  "
                      f"FAIL  ({dt:.1f}s)")
                continue

            # 저장
            out_name = job.batch_id.replace(":", "_") + "-graph.json"
            out_path = OUT_DIR / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            idx_out.write(json.dumps({
                "batch_id": job.batch_id,
                "status": "ok",
                "n_nodes": len(data.get("nodes", [])),
                "n_edges": len(data.get("edges", [])),
                "warns": warns, "dt": round(dt, 2),
            }, ensure_ascii=False) + "\n")
            idx_out.flush()
            n_ok += 1
            if i % 10 == 0 or i == len(jobs):
                elapsed = time.time() - t_start
                rate = i / max(1, elapsed)
                eta = (len(jobs) - i) / max(0.01, rate) / 60
                print(f"  [{i:>4}/{len(jobs)}] {job.batch_id}  "
                      f"OK  nodes={len(data.get('nodes', []))} "
                      f"edges={len(data.get('edges', []))}  "
                      f"({dt:.1f}s · {rate*60:.1f}/min · ETA {eta:.0f}m)")

    print()
    print(f"[done] ok={n_ok}  fail={n_err}  skipped={n_skip}  "
          f"total elapsed={(time.time()-t_start)/60:.1f}m")
    print(f"[done] outputs: {OUT_DIR}")
    return 0 if n_err == 0 else 2


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="시작 날짜 YYYY-MM-DD")
    ap.add_argument("--end", help="종료 날짜 YYYY-MM-DD")
    ap.add_argument("--limit", type=int, help="처리 최대 페이지 수")
    ap.add_argument("--smoke", action="store_true",
                    help="50페이지 스모크 테스트 (1917-04 중심)")
    ap.add_argument("--dry-run", action="store_true",
                    help="모델 로드 없이 작업 계획만 출력")
    args = ap.parse_args()
    return process(args.start, args.end, args.limit, args.smoke, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())