from taxmate.corpus import DOCS, load_chunks

CHUNKS = load_chunks()


def test_all_seven_docs_present():
    assert {c.doc_id for c in CHUNKS} == {d["doc_id"] for d in DOCS} and len(DOCS) == 7


def test_rmc50_one_chunk_per_question():
    rmc = [c for c in CHUNKS if c.doc_id == "RMC-50-2018"]
    assert len(rmc) == 62 and {c.section for c in rmc} == {f"Q{i}" for i in range(1, 63)}
    assert "May 15" in next(c.text for c in rmc if c.section == "Q19")


def test_rmc50_pdf_not_used():
    assert not any("REI)Ut}LIC" in c.text or "incluclecl" in c.text for c in CHUNKS)


def test_chunks_have_metadata_and_text():
    assert all(c.title and c.issued and c.section and len(c.text.strip()) >= 40 for c in CHUNKS)


def test_corpus_files_exist_for_every_doc():
    from pathlib import Path
    assert all((Path("corpus") / f"{d['doc_id']}.jsonl").is_file() for d in DOCS)


def test_loads_from_corpus_without_source_files(tmp_path):
    import shutil
    shutil.copytree("corpus", tmp_path / "corpus")  # no bir-data/ in tmp_path
    chunks = load_chunks(tmp_path)
    assert len(chunks) == len(CHUNKS) and {c.doc_id for c in chunks} == {d["doc_id"] for d in DOCS}


def _records(doc_id):
    import json
    from pathlib import Path
    return [json.loads(l) for l in (Path("corpus") / f"{doc_id}.jsonl").read_text(encoding="utf-8").splitlines()]


def test_page_footers_stripped():
    import re
    footer = re.compile(r"^\s*\d+\s*/\s*\d+\s*$", re.MULTILINE)
    assert not any(footer.search(r["text"]) for d in DOCS for r in _records(d["doc_id"]))


def test_no_intra_word_spaces_from_pdf_extraction():
    text = "\n".join(r["text"] for d in DOCS for r in _records(d["doc_id"]))
    for broken in ["s hall", "incom e", "c hosen", "i mplements", "investm ent", "thre shold"]:
        assert broken not in text, broken


def test_rr11_p32_has_no_phantom_list_marker():
    p32 = next(r["text"] for r in _records("RR-11-2018") if r["section"] == "p32")
    import re
    assert not re.search(r"(?<![\d,.])6\.\s", p32)  # standalone "6." marker, not e.g. "2,166.73"


def test_heading_split_gives_one_chunk_per_section():
    from taxmate.corpus import _split_on_headings
    text = "Intro line\nFile and Pay Anywhere\nFile with any AAB.\nEase of Registration\nNo more P500 fee."
    secs = _split_on_headings(text, ["File and Pay Anywhere", "Ease of Registration"])
    assert secs == [("Intro", "Intro line"),
                    ("File and Pay Anywhere", "File and Pay Anywhere\nFile with any AAB."),
                    ("Ease of Registration", "Ease of Registration\nNo more P500 fee.")]


def test_eopt_flyer_file_and_pay_anywhere_is_its_own_chunk():
    eopt = [c for c in load_chunks() if c.doc_id == "EOPT-FLYER"]
    anywhere = [c for c in eopt if c.section == "File and Pay Anywhere"]
    assert len(anywhere) == 1
    assert "Authorized Tax Software Provider" in anywhere[0].text
    assert "Classification" not in anywhere[0].text and "Ease of Registration" not in anywhere[0].text
