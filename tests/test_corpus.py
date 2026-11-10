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
