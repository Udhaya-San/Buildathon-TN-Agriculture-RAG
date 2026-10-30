"""Retrieve relevant chunks and generate a grounded answer."""

from langchain_openai import ChatOpenAI

import config


def get_llm():
    return ChatOpenAI(model=config.CHAT_MODEL, temperature=0)


def retrieve(vectorstore, query: str, k: int = config.TOP_K, schemes=None):
    """Return the k most relevant chunks using MMR (relevant but not repetitive).

    schemes: optional list of scheme names. Only chunks tagged with at least
    one of them are considered.
    """
    metadata_filter = None
    if schemes:
        wanted = set(schemes)
        metadata_filter = lambda md: bool(wanted & set(md.get("schemes", [])))
    return vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=config.MMR_FETCH_K,
        lambda_mult=config.MMR_LAMBDA,
        filter=metadata_filter,
    )


def format_context(docs) -> str:
    parts = []
    for doc in docs:
        parts.append(f"[Page {doc.metadata.get('page_number')}]\n{doc.page_content}")
    return "\n\n".join(parts)


def format_history(history) -> str:
    """history is a list of {"question": ..., "answer": ...}; only the last few are used."""
    recent = history[-config.HISTORY_TURNS:] if history else []
    return "\n".join(f"User: {turn['question']}\nAssistant: {turn['answer']}" for turn in recent)


def build_prompt(question: str, context: str, history=None) -> str:
    chat = format_history(history or [])
    chat_block = f"Earlier conversation (only to understand follow-up questions):\n{chat}\n\n" if chat else ""
    return f"""You are a Tamil Nadu Agriculture Schemes Assistant.

Rules:
1. Answer only from the context below. Do not use outside knowledge.
2. Do not make assumptions.
3. Mention the scheme name when the context gives it.
4. End the answer with the pages you used, like: (Pages: 12, 14).
5. If the context does not contain the answer, reply exactly:
{config.NOT_FOUND_MESSAGE}

{chat_block}Context:
{context}

Question:
{question}

Answer:
"""


def answer_question(vectorstore, llm, question: str, history=None, schemes=None) -> dict:
    """Run the full RAG step. Returns answer, retrieved chunks and source pages."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Question is empty.")

    docs = retrieve(vectorstore, question, schemes=schemes)
    if not docs:
        # Decided by code, not by the model: nothing retrieved means nothing to answer from.
        return {"answer": config.NOT_FOUND_MESSAGE, "docs": [], "pages": [], "found": False}

    prompt = build_prompt(question, format_context(docs), history)
    answer = llm.invoke(prompt).content.strip()
    found = config.NOT_FOUND_MESSAGE.lower() not in answer.lower()
    pages = sorted({d.metadata.get("page_number") for d in docs}) if found else []
    return {"answer": answer, "docs": docs, "pages": pages, "found": found}
