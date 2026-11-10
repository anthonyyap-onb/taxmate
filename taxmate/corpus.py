"""BIR corpus: extract source documents to committed text files, then chunk them.

Source documents live in bir-data/. Run `python -m taxmate.corpus build` to extract their text into
corpus/<doc_id>.jsonl; the app reads only corpus/ at runtime (no PDF parsing on startup)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf

from taxmate.types import Chunk

DOCS: list[dict] = [
    {"doc_id": "RR-8-2018", "source": "bir-data/RR 8-2018.pdf",
     "title": "Revenue Regulations No. 8-2018 (digest)", "issued": "2018-02-20"},
    {"doc_id": "RMC-50-2018", "source": "bir-data/transcribed/RMC_No_50-2018.md",
     "title": "Revenue Memorandum Circular No. 50-2018", "issued": "2018-05-11"},
    {"doc_id": "RR-11-2018", "source": "bir-data/RR No. 11-2018.pdf",
     "title": "Revenue Regulations No. 11-2018", "issued": "2018-01-31"},
    {"doc_id": "1701Q-GUIDE", "source": "bir-data/Guidelines for BIR Form 1701Q.pdf",
     "title": "Guidelines for BIR Form 1701Q", "issued": "2018-01"},
    {"doc_id": "1701MS-GUIDE", "source": "bir-data/Guidelines for BIR Form 1701-MS.pdf",
     "title": "Guidelines for BIR Form 1701-MS", "issued": "2024-08"},
    {"doc_id": "2551Q-GUIDE", "source": "bir-data/Guidelines for BIR Form 2551Q.pdf",
     "title": "Guidelines for BIR Form 2551Q", "issued": "2018-01"},
    {"doc_id": "EOPT-FLYER", "source": "bir-data/EOPT Flyer.pdf",
     "title": "Ease of Paying Taxes (EOPT) Act Flyer", "issued": "2024"},
]

MIN_CHARS = 40
_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
_QA_HEADING = re.compile(r"^## Q(\d+)\s*$", re.MULTILINE)


CORPUS_DIR = "corpus"


def _split_qa(text: str) -> list[dict]:
    matches = list(_QA_HEADING.finditer(text))
    records = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].split("\n---\n")[0].strip()  # drop trailing signature block
        records.append({"section": f"Q{m.group(1)}", "text": body})
    return records


_PAGE_FOOTER = re.compile(r"^[ \t]*\d+[ \t]*/[ \t]*\d+[ \t]*\n?", re.MULTILINE)  # e.g. "28 / 47"


def _same_row(a: tuple, b: tuple) -> bool:
    """True if bbox b sits on the same visual row as a, to its right (vertical overlap > half the smaller height)."""
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    return b[0] >= a[2] - 1 and overlap > 0.5 * min(a[3] - a[1], b[3] - b[1])


def _page_text(page: pymupdf.Page) -> str:
    """Page text in PyMuPDF's native reading order (column-correct), but with consecutive lines that share a
    baseline (table cells, computation rows) joined into one line instead of one cell per line."""
    rows: list[list] = []  # each row: [bbox, text]
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text:
                continue
            if rows and _same_row(rows[-1][0], line["bbox"]):
                prev = rows[-1]
                prev[0] = (prev[0][0], min(prev[0][1], line["bbox"][1]), line["bbox"][2], max(prev[0][3], line["bbox"][3]))
                prev[1] += " " + text
            else:
                rows.append([line["bbox"], text])
    return _PAGE_FOOTER.sub("", "\n".join(text for _, text in rows))


def _extract(path: Path) -> list[dict]:
    if path.suffix == ".md":
        return _split_qa(path.read_text(encoding="utf-8"))
    with pymupdf.open(path) as pdf:
        return [{"section": f"p{i + 1}", "text": _page_text(page)} for i, page in enumerate(pdf)]


def build_corpus(root: Path = Path(".")) -> dict[str, int]:
    """Extract every source document into corpus/<doc_id>.jsonl. Returns records written per doc."""
    out_dir = Path(root) / CORPUS_DIR
    out_dir.mkdir(exist_ok=True)
    counts = {}
    for doc in DOCS:
        records = _extract(Path(root) / doc["source"])
        with (out_dir / f"{doc['doc_id']}.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[doc["doc_id"]] = len(records)
    return counts


def _read_records(root: Path, doc_id: str) -> list[dict]:
    path = Path(root) / CORPUS_DIR / f"{doc_id}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _page_chunks(doc: dict, pages: list[dict]) -> list[Chunk]:
    full, starts = "", []  # starts: (char offset, page section)
    for page in pages:
        starts.append((len(full), page["section"]))
        full += page["text"] + "\n"
    chunks, cursor = [], 0
    for i, piece in enumerate(_SPLITTER.split_text(full)):
        pos = full.find(piece[:80], cursor)
        if pos == -1:
            pos = cursor
        cursor = pos
        section = [sec for off, sec in starts if off <= pos][-1] if starts else "p1"
        chunks.append(Chunk(f"{doc['doc_id']}#{section}#{i}", doc["doc_id"], doc["title"], doc["issued"], section, piece))
    return chunks


def load_chunks(root: Path = Path(".")) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in DOCS:
        records = _read_records(Path(root), doc["doc_id"])
        if doc["source"].endswith(".md"):
            chunks = [Chunk(f"{doc['doc_id']}#{r['section']}#0", doc["doc_id"], doc["title"], doc["issued"],
                            r["section"], r["text"]) for r in records]
        else:
            chunks = _page_chunks(doc, records)
        out.extend(c for c in chunks if len(c.text.strip()) >= MIN_CHARS)
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        for doc_id, n in build_corpus().items():
            print(f"{doc_id}: {n} records")
    else:
        print("usage: python -m taxmate.corpus build")
