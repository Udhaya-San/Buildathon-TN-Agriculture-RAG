# Tamil Nadu Agriculture Schemes Assistant (RAG)

A Streamlit app that answers questions about Tamil Nadu agriculture schemes from one
PDF, *Tamil Nadu Agricultural Schemes Made Simple, Vol II* (182 pages, English and Tamil).
It finds the most relevant passages in a FAISS index and asks an OpenAI model to answer
**only from those passages**, with page numbers.

## Quick start (from a fresh clone)

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
python -m pip install -r requirements.txt
copy .env.example .env           # macOS/Linux: cp .env.example .env   then add your OPENAI_API_KEY
python build_index.py            # builds the search index from the PDF (one time, about a minute)
streamlit run app.py
```

`build_index.py` is required: the index is not stored in Git, it is generated from the PDF.

## How it works

```
PDF ──> clean (keep English lines) ──> split into chunks ──> embed ──> FAISS index
                                                                          │
question ──> MMR search (optional scheme filter) ──> top 4 chunks ──> prompt ──> answer + pages
```

| Step | File | What happens |
|---|---|---|
| Load and clean | `pdf_processing.py` | Reads the PDF, drops cover/Tamil/garbled lines |
| Chunk and tag | `pdf_processing.py` | 800-character chunks, 150 overlap; each chunk gets its page, id and scheme tags |
| Index | `vector_store.py`, `build_index.py` | Embeds chunks, saves FAISS plus `index_meta.json` with the real counts |
| Retrieve and answer | `rag_pipeline.py` | MMR search, grounded prompt with chat history, answer with pages |
| Interface | `app.py`, `ui_styles.py` | Chat, scheme filter, source pages, retrieved chunks |
| Settings | `config.py` | Models, chunk size, retrieval numbers, scheme list |
| Evaluation | `evaluate.py`, `eval/` | Hit-rate and MRR on a hand-labelled question set |
| Tests | `tests/test_rag.py` | 13 offline tests |

## Design choices

- **English text only.** The Tamil text in this PDF uses an old font encoding and extracts as
  scrambled symbols. Indexing it would add noise, so only the English lines are indexed
  (103 of 182 pages have enough English text; the rest are covers and Tamil-only pages).
- **MMR retrieval.** Maximal Marginal Relevance picks chunks that are relevant *and* different
  from each other, so the answer is not built from four copies of the same paragraph.
- **Scheme tags.** Each chunk is tagged with the schemes it mentions (for example eNAM, PM-KISAN).
  The sidebar can limit a search to chosen schemes.
- **Real statistics.** Page and chunk counts shown in the app are read from `index_meta.json`,
  written when the index is built. Nothing is typed in by hand.
- **Grounded answers.** The prompt allows only the supplied context. If the answer is not there
  the app replies "Information not available in the Agriculture Schemes document."
- **Page numbers** are PDF viewer pages. The book's own printed page number is the PDF page minus 15.

## Evaluation

`eval/eval_questions.json` has 27 questions: 24 whose answers sit on known pages, and 3
out-of-scope questions that must be refused.

```bash
python evaluate.py                 # retrieval: hit-rate@4 and MRR@4
python evaluate.py --with-answers  # also checks answers contain the expected facts, and refusals
```

Results are written to `eval/eval_results.csv` and `eval/eval_report.json`, together with the
settings used. Add the numbers from your own run below.

| Date | Hit-rate@4 | MRR@4 | Keyword match | Refusal rate |
|---|---|---|---|---|
| 30-10-2026 | 96% | 0.91 | 88% | 100% |

## Tests

```bash
python tests/test_rag.py
```

The tests need no API key. They use fake embeddings and a fake model, and they also check that
every labelled page and keyword in the evaluation set really appears in the PDF.

## Limitations

- Tamil text is not indexed, so Tamil questions rely on the multilingual embedding model and may
  retrieve less accurately than English ones.
- Answers cover only what is written in this one PDF. It is a guide, not an official order, and
  scheme details change; check the department website before applying.
- Retrieval is vector search with MMR. There is no keyword (BM25) search or reranking, so exact
  codes or very short queries can miss.
- The scheme tags come from a fixed keyword list in `config.py`, so a scheme not on the list is untagged.
- The FAISS index is saved with pickle. Load only an index you built yourself with `build_index.py`.
- The answer quality depends on the OpenAI model; a wrong answer is still possible, so check the cited pages.

## Learning steps

`learning_steps/` keeps the original step-by-step scripts (`v1` to `v11`) used to learn each part
of RAG: loading, splitting, embeddings, FAISS, retriever, prompt and UI.
