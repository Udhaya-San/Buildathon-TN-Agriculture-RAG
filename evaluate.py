"""Measure how good the search and the answers are, using eval/eval_questions.json.

    python evaluate.py                 # retrieval only (cheap: embeddings only)
    python evaluate.py --with-answers  # also generate answers and check them

Metrics
  hit-rate : share of questions where a correct page is among the retrieved chunks
  MRR      : mean of 1/rank of the first correct page (1.0 = always ranked first)
  keyword  : share of answers that contain every expected keyword
  refusal  : share of out-of-scope questions answered with the not-available reply
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

import config
from rag_pipeline import answer_question, get_llm, retrieve
from vector_store import get_embeddings, load_index

QUESTIONS_PATH = config.BASE_DIR / "eval" / "eval_questions.json"
RESULTS_CSV = config.BASE_DIR / "eval" / "eval_results.csv"
REPORT_JSON = config.BASE_DIR / "eval" / "eval_report.json"


def load_questions(path: Path = QUESTIONS_PATH) -> list:
    return json.loads(Path(path).read_text(encoding="utf-8"))["questions"]


def reciprocal_rank(retrieved_pages: list, expected_pages: list) -> float:
    """1/rank of the first retrieved page that is correct, else 0."""
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected_pages:
            return 1.0 / rank
    return 0.0


def keywords_present(answer: str, keywords: list) -> bool:
    text = answer.lower()
    return all(word.lower() in text for word in keywords)


def run_evaluation(vectorstore, questions: list, llm=None) -> dict:
    """Run every question. llm=None means retrieval-only."""
    rows = []
    for item in questions:
        out_of_scope = item.get("out_of_scope", False)
        docs = retrieve(vectorstore, item["question"])
        pages = [d.metadata.get("page_number") for d in docs]
        row = {
            "id": item["id"],
            "question": item["question"],
            "out_of_scope": out_of_scope,
            "retrieved_pages": pages,
            "expected_pages": item["expected_pages"],
            "rr": None if out_of_scope else reciprocal_rank(pages, item["expected_pages"]),
            "answer": "",
            "keywords_ok": None,
            "refused": None,
        }
        if llm is not None:
            result = answer_question(vectorstore, llm, item["question"])
            row["answer"] = result["answer"]
            if out_of_scope:
                row["refused"] = not result["found"]
            else:
                row["keywords_ok"] = keywords_present(result["answer"], item["keywords"])
        rows.append(row)

    scored = [r for r in rows if not r["out_of_scope"]]
    summary = {
        "questions": len(rows),
        "in_scope": len(scored),
        "hit_rate": sum(1 for r in scored if r["rr"] > 0) / len(scored) if scored else 0.0,
        "mrr": sum(r["rr"] for r in scored) / len(scored) if scored else 0.0,
        "top_k": config.TOP_K,
    }
    if llm is not None:
        with_kw = [r for r in scored if r["keywords_ok"] is not None]
        refusals = [r for r in rows if r["out_of_scope"]]
        summary["keyword_rate"] = sum(r["keywords_ok"] for r in with_kw) / len(with_kw) if with_kw else 0.0
        summary["refusal_rate"] = sum(r["refused"] for r in refusals) / len(refusals) if refusals else 0.0
    return {"summary": summary, "rows": rows}


def save_results(result: dict, index_meta: dict) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "question", "out_of_scope", "expected_pages", "retrieved_pages", "rr", "keywords_ok", "refused", "answer"])
        for r in result["rows"]:
            writer.writerow([r["id"], r["question"], r["out_of_scope"], r["expected_pages"],
                             r["retrieved_pages"], r["rr"], r["keywords_ok"], r["refused"], r["answer"]])
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": result["summary"],
        "index": index_meta,
        "settings": {"top_k": config.TOP_K, "mmr_fetch_k": config.MMR_FETCH_K,
                     "mmr_lambda": config.MMR_LAMBDA, "chat_model": config.CHAT_MODEL},
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers.")
    parser.add_argument("--with-answers", action="store_true", help="also generate and check answers")
    args = parser.parse_args()

    store, meta = load_index(embeddings=get_embeddings())
    result = run_evaluation(store, load_questions(), llm=get_llm() if args.with_answers else None)
    save_results(result, meta)

    s = result["summary"]
    print(f"Questions: {s['questions']} ({s['in_scope']} in scope)")
    print(f"Hit-rate@{s['top_k']}: {s['hit_rate']:.0%}")
    print(f"MRR@{s['top_k']}     : {s['mrr']:.2f}")
    if "keyword_rate" in s:
        print(f"Keyword match   : {s['keyword_rate']:.0%}")
        print(f"Refusal rate    : {s['refusal_rate']:.0%}")
    print(f"Details saved to {RESULTS_CSV.relative_to(config.BASE_DIR)}")
    misses = [r for r in result["rows"] if not r["out_of_scope"] and r["rr"] == 0]
    for r in misses:
        print(f"  MISS  Q{r['id']}: {r['question']}  expected {r['expected_pages']} got {r['retrieved_pages']}")


if __name__ == "__main__":
    main()
