"""Step 1 of the pipeline: read the PDF, clean it, split it into chunks.

This PDF is bilingual. Its Tamil text uses an old font encoding, so the text
extractor returns scrambled symbols for it. Those lines would only add noise
to the embeddings, so cleaning keeps the English lines and drops the rest.
"""

import hashlib
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

_TAMIL_OR_BROKEN = re.compile(r"[஀-௿�]")


def clean_page_text(text: str) -> str:
    """Keep English lines, drop Tamil/garbled lines and empty lines."""
    kept = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        letters = sum(ch.isalpha() for ch in line)
        if letters < 3:
            # No real words: keep only if it carries a number (e.g. a table row)
            if re.search(r"\d", line):
                kept.append(line)
            continue
        ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in line)
        if ascii_letters / letters >= 0.9 and not _TAMIL_OR_BROKEN.search(line):
            kept.append(line)
    return "\n".join(kept)


def tag_schemes(text: str) -> list:
    """Return the names of the schemes mentioned in a piece of text."""
    found = []
    for name, patterns in config.SCHEME_PATTERNS.items():
        if any(re.search(p, text, flags=re.IGNORECASE) for p in patterns):
            found.append(name)
    return found


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pdf_pages(pdf_path: Path = config.PDF_PATH):
    """Load the PDF and return (cleaned_pages, total_pdf_pages).

    Pages that have almost no English text after cleaning are skipped.
    metadata["page"] is the 0-based PDF page index, metadata["page_number"]
    is the 1-based number you see in a PDF viewer.
    """
    raw_pages = PyPDFLoader(str(pdf_path)).load()
    pages = []
    for doc in raw_pages:
        text = clean_page_text(doc.page_content)
        if len(text) < config.MIN_PAGE_CHARS:
            continue
        index = doc.metadata.get("page", 0)
        pages.append(
            Document(
                page_content=text,
                metadata={
                    "source": Path(pdf_path).name,
                    "page": index,
                    "page_number": index + 1,
                },
            )
        )
    return pages, len(raw_pages)


def split_pages(pages):
    """Split pages into overlapping chunks and attach scheme tags."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    for number, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = number
        chunk.metadata["schemes"] = tag_schemes(chunk.page_content)
    return chunks
