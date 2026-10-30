"""Offline tests: no OpenAI key or internet needed (fake embeddings and a fake LLM).

Run from the project folder:   python tests/test_rag.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import config
import evaluate
import rag_pipeline
import vector_store
from pdf_processing import clean_page_text, load_pdf_pages, split_pages, tag_schemes

_EMB = DeterministicFakeEmbedding(size=64)
_CACHE = {}


def _index():
    """Build one small index for the whole test run."""
    if "store" not in _CACHE:
        folder = Path(tempfile.mkdtemp())
        meta = vector_store.build_index(embeddings=_EMB, index_dir=folder)
        store, _ = vector_store.load_index(embeddings=_EMB, index_dir=folder)
        _CACHE.update(store=store, meta=meta, folder=folder)
    return _CACHE["store"], _CACHE["meta"], _CACHE["folder"]


def test_cleaning_keeps_english_and_drops_tamil_or_garbled():
    raw = "Eligibility: farmers aged 18-65\nதமிழ்நாடு அரசு திட்டம்\nÖ.¼k.g.ï.-ïòDA Öª©ØÃòÂz\n\n"
    cleaned = clean_page_text(raw)
    assert "Eligibility" in cleaned
    assert "தமிழ்" not in cleaned
    assert "Ö" not in cleaned


def test_scheme_tagging():
    assert "eNAM" in tag_schemes("The electronic National Agriculture Market (eNAM) helps farmers")
    assert "PM-KISAN" in tag_schemes("Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)")
    assert tag_schemes("Nothing about any scheme here") == []


def test_chunks_carry_metadata():
    pages, total = load_pdf_pages()
    chunks = split_pages(pages[:5])
    assert total > len(pages) > 0
    assert all({"page", "page_number", "chunk_id", "schemes"} <= set(c.metadata) for c in chunks)
    assert all(c.metadata["page_number"] == c.metadata["page"] + 1 for c in chunks)


def test_build_index_reports_real_numbers():
    store, meta, folder = _index()
    assert meta["chunks"] == store.index.ntotal  # statistic equals what is really stored
    assert meta["indexed_pages"] < meta["pdf_pages"]
    assert (folder / "index.faiss").exists() and (folder / "index_meta.json").exists()


def test_missing_index_gives_clear_error():
    try:
        vector_store.load_index(embeddings=_EMB, index_dir=Path(tempfile.mkdtemp()) / "nope")
    except vector_store.IndexNotFoundError as error:
        assert "build_index.py" in str(error)
        return
    raise AssertionError("expected IndexNotFoundError")


def test_scheme_filter_only_returns_tagged_chunks():
    store, _, _ = _index()
    docs = rag_pipeline.retrieve(store, "subsidy", schemes=["eNAM"])
    assert docs and all("eNAM" in d.metadata["schemes"] for d in docs)


def test_empty_question_is_rejected():
    store, _, _ = _index()
    try:
        rag_pipeline.answer_question(store, FakeListChatModel(responses=["x"]), "   ")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_answer_with_and_without_evidence():
    store, _, _ = _index()
    good = rag_pipeline.answer_question(store, FakeListChatModel(responses=["An answer (Pages: 1)"]), "What is eNAM?")
    assert good["found"] and good["pages"] == sorted(good["pages"]) and good["docs"]
    none = rag_pipeline.answer_question(store, FakeListChatModel(responses=[config.NOT_FOUND_MESSAGE]), "Who won the world cup?")
    assert not none["found"] and none["pages"] == []


def test_prompt_contains_history_and_rules():
    history = [{"question": "What is eNAM?", "answer": "An online market."}]
    prompt = rag_pipeline.build_prompt("And its benefits?", "[Page 1] text", history)
    assert "An online market." in prompt and config.NOT_FOUND_MESSAGE in prompt


def test_reciprocal_rank():
    assert evaluate.reciprocal_rank([5, 9, 7], [9]) == 0.5
    assert evaluate.reciprocal_rank([5, 9], [5]) == 1.0
    assert evaluate.reciprocal_rank([1, 2], [3]) == 0.0


def test_eval_questions_match_the_pdf():
    """Every labelled page must exist in the index and contain its keywords' evidence."""
    pages, _ = load_pdf_pages()
    text_by_page = {d.metadata["page_number"]: d.page_content.lower() for d in pages}
    questions = evaluate.load_questions()
    assert len(questions) >= 25 and len({q["id"] for q in questions}) == len(questions)
    for q in questions:
        if q.get("out_of_scope"):
            continue
        assert q["expected_pages"], f"Q{q['id']} has no expected page"
        for page in q["expected_pages"]:
            assert page in text_by_page, f"Q{q['id']}: page {page} is not indexed"
        for word in q["keywords"]:
            assert any(word.lower() in text_by_page[p] for p in q["expected_pages"]), \
                f"Q{q['id']}: keyword {word!r} not found on page(s) {q['expected_pages']}"


def test_evaluation_runs_end_to_end_offline():
    store, _, _ = _index()
    llm = FakeListChatModel(responses=["answer"] * 40)
    result = evaluate.run_evaluation(store, evaluate.load_questions()[:3] + evaluate.load_questions()[-1:], llm=llm)
    s = result["summary"]
    assert s["questions"] == 4 and s["in_scope"] == 3
    assert 0.0 <= s["hit_rate"] <= 1.0 and 0.0 <= s["mrr"] <= 1.0


def test_embeddings_do_not_need_tokenizer_download():
    import os
    os.environ.setdefault("OPENAI_API_KEY", "sk-test")
    assert vector_store.get_embeddings().check_embedding_ctx_length is False


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, func in tests:
        try:
            func()
            print(f"PASS  {name}")
        except Exception as error:  # report every failure, not only the first
            failed += 1
            print(f"FAIL  {name}: {error}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
