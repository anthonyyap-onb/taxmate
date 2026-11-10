"""Load the BIR corpus into metadata-rich chunks."""
from __future__ import annotations

import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from taxmate.types import Chunk

DOCS: list[dict] = [
    {"doc_id": "RR-8-2018", "path": "bir-data/RR 8-2018.pdf",
     "title": "Revenue Regulations No. 8-2018 (digest)", "issued": "2018-02-20"},
    {"doc_id": "RMC-50-2018", "path": "bir-data/transcribed/RMC_No_50-2018.md",
     "title": "Revenue Memorandum Circular No. 50-2018", "issued": "2018-05-11"},
    {"doc_id": "RR-11-2018", "path": "bir-data/RR No. 11-2018.pdf",
     "title": "Revenue Regulations No. 11-2018", "issued": "2018-01-31"},
    {"doc_id": "1701Q-GUIDE", "path": "bir-data/Guidelines for BIR Form 1701Q.pdf",
     "title": "Guidelines for BIR Form 1701Q", "issued": "2018-01"},
    {"doc_id": "1701MS-GUIDE", "path": "bir-data/Guidelines for BIR Form 1701-MS.pdf",
     "title": "Guidelines for BIR Form 1701-MS", "issued": "2024-08"},
    {"doc_id": "2551Q-GUIDE", "path": "bir-data/Guidelines for BIR Form 2551Q.pdf",
     "title": "Guidelines for BIR Form 2551Q", "issued": "2018-01"},
    {"doc_id": "EOPT-FLYER", "path": "bir-data/EOPT Flyer.pdf",
     "title": "Ease of Paying Taxes (EOPT) Act Flyer", "issued": "2024"},
]

MIN_CHARS = 40
_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
_QA_HEADING = re.compile(r"^## Q(\d+)\s*$", re.MULTILINE)


def _qa_chunks(doc: dict, text: str) -> list[Chunk]:
    matches = list(_QA_HEADING.finditer(text))
    chunks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].split("\n---\n")[0].strip()  # drop trailing signature block
        section = f"Q{m.group(1)}"
        chunks.append(Chunk(f"{doc['doc_id']}#{section}#0", doc["doc_id"], doc["title"], doc["issued"], section, body))
    return chunks


def _pdf_chunks(doc: dict, path: Path) -> list[Chunk]:
    pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(path).pages)]
    full, starts = "", []  # starts: (char offset, page number)
    for page_no, text in pages:
        starts.append((len(full), page_no))
        full += text + "\n"
    chunks, cursor = [], 0
    for i, piece in enumerate(_SPLITTER.split_text(full)):
        pos = full.find(piece[:80], cursor)
        if pos == -1:
            pos = cursor
        cursor = pos
        page = max(p for off, p in starts if off <= pos) if starts else 1
        section = f"p{page}"
        chunks.append(Chunk(f"{doc['doc_id']}#{section}#{i}", doc["doc_id"], doc["title"], doc["issued"], section, piece))
    return chunks


def load_chunks(root: Path = Path(".")) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in DOCS:
        path = Path(root) / doc["path"]
        if path.suffix == ".md":
            chunks = _qa_chunks(doc, path.read_text(encoding="utf-8"))
        else:
            chunks = _pdf_chunks(doc, path)
        out.extend(c for c in chunks if len(c.text.strip()) >= MIN_CHARS)
    return out
