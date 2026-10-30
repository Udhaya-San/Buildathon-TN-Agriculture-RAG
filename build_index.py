"""Build the search index from the PDF.  Run this once after cloning:

    python build_index.py
"""

import argparse
import os
import sys

from dotenv import load_dotenv

import config
from vector_store import build_index


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build the FAISS index from the PDF.")
    parser.add_argument("--pdf", default=str(config.PDF_PATH), help="path to the PDF")
    parser.add_argument("--index-dir", default=str(config.INDEX_DIR), help="where to save the index")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    print(f"Reading {args.pdf} ...")
    meta = build_index(pdf_path=args.pdf, index_dir=args.index_dir)
    print("Index built.")
    print(f"  PDF pages        : {meta['pdf_pages']}")
    print(f"  Pages indexed    : {meta['indexed_pages']} (English text only)")
    print(f"  Chunks           : {meta['chunks']}")
    print(f"  Chunks with a scheme tag : {meta['chunks_with_scheme_tag']}")
    print(f"  Embedding model  : {meta['embedding_model']}")
    print(f"  Saved to         : {args.index_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
