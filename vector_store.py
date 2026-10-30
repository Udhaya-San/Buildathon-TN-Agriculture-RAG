"""Build, save and load the FAISS index, and report real statistics about it."""

import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

import config
from pdf_processing import file_sha256, load_pdf_pages, split_pages


class IndexNotFoundError(RuntimeError):
    """Raised when the index has not been built yet."""


def get_embeddings():
    # check_embedding_ctx_length=False stops LangChain from downloading a tokenizer
    # file (tiktoken) from the internet. That download fails on networks that
    # intercept HTTPS. It is safe here because every chunk is only a few hundred
    # characters, far below the embedding model's input limit.
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL, check_embedding_ctx_length=False)


def build_index(embeddings=None, pdf_path: Path = config.PDF_PATH, index_dir: Path = config.INDEX_DIR):
    """Read the PDF, embed every chunk and save the index. Returns its statistics."""
    pdf_path, index_dir = Path(pdf_path), Path(index_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages, total_pages = load_pdf_pages(pdf_path)
    chunks = split_pages(pages)
    if not chunks:
        raise ValueError("No text could be extracted from the PDF.")

    embeddings = embeddings or get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))

    tagged = sum(1 for c in chunks if c.metadata["schemes"])
    meta = {
        "pdf_file": pdf_path.name,
        "pdf_sha256": file_sha256(pdf_path),
        "pdf_pages": total_pages,
        "indexed_pages": len(pages),
        "chunks": len(chunks),
        "chunks_with_scheme_tag": tagged,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "embedding_model": config.EMBEDDING_MODEL,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (index_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def read_index_meta(index_dir: Path = config.INDEX_DIR) -> dict:
    path = Path(index_dir) / "index_meta.json"
    if not path.exists():
        raise IndexNotFoundError(
            "The search index has not been built yet. Run:  python build_index.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_index(embeddings=None, index_dir: Path = config.INDEX_DIR):
    """Load the saved index. Raises IndexNotFoundError with a clear message if missing."""
    index_dir = Path(index_dir)
    meta = read_index_meta(index_dir)  # also proves the folder exists
    if not (index_dir / "index.faiss").exists():
        raise IndexNotFoundError(
            "The search index files are missing. Run:  python build_index.py"
        )
    embeddings = embeddings or get_embeddings()
    # The index is pickled by FAISS, so only load an index you built yourself
    # with build_index.py from the included PDF.
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True), meta
