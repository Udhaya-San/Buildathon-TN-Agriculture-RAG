"""Streamlit app: ask questions about Tamil Nadu agriculture schemes.

Run:  streamlit run app.py     (build the index first: python build_index.py)
"""

import os

import streamlit as st
from dotenv import load_dotenv

import config
import ui_styles
from rag_pipeline import answer_question, get_llm
from vector_store import IndexNotFoundError, get_embeddings, load_index

load_dotenv()

st.set_page_config(page_title="TN Agriculture Schemes Assistant", page_icon="🌾", layout="wide")
st.markdown(ui_styles.CSS, unsafe_allow_html=True)
st.markdown(ui_styles.GOV_HEADER, unsafe_allow_html=True)
st.markdown(ui_styles.TITLE, unsafe_allow_html=True)

# ---- Start-up checks: show a clear message instead of a crash ---------------
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is missing. Copy `.env.example` to `.env`, add your key and restart the app.")
    st.stop()


@st.cache_resource
def load_components():
    store, meta = load_index(embeddings=get_embeddings())
    return store, get_llm(), meta


try:
    vectorstore, llm, index_meta = load_components()
except IndexNotFoundError as error:
    st.error(str(error))
    st.stop()

# ---- Knowledge base numbers come from the saved index, not typed in ---------
st.markdown("## 📊 Knowledge Base")
col1, col2, col3, col4 = st.columns(4)
col1.metric("📄 PDF pages", index_meta["pdf_pages"])
col2.metric("📖 Pages indexed", index_meta["indexed_pages"])
col3.metric("📚 Chunks", index_meta["chunks"])
col4.metric("⚡ Embeddings", index_meta["embedding_model"])
st.caption(
    "Only the English text of the bilingual PDF is indexed, because its Tamil text "
    "does not extract cleanly. Index built at " + index_meta["built_at"] + "."
)
st.divider()

# ---- Sidebar ---------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

with st.sidebar:
    try:
        st.image("assets/tn_logo.png", width=120)
    except Exception:
        pass
    st.title("🌾 Agriculture Portal")
    selected_schemes = st.multiselect(
        "Limit search to scheme(s)",
        options=list(config.SCHEME_PATTERNS),
        help="Optional. Leave empty to search the whole document.",
    )
    st.markdown("**Try asking**")
    examples = [
        "What is eNAM?",
        "What are the organic farming schemes?",
        "Tell me about the Agriculture Infrastructure Fund",
        "What awards are given to farmers?",
    ]
    for example in examples:
        if st.button(example, key=f"ex_{example}"):
            st.session_state["pending_question"] = example
    if st.button("🗑️ Clear chat"):
        st.session_state["history"] = []
        st.rerun()

# ---- Chat ------------------------------------------------------------------
for turn in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        if turn["pages"]:
            st.caption("Source pages: " + ", ".join(str(p) for p in turn["pages"]))

typed = st.chat_input("Ask about a Tamil Nadu agriculture scheme...")
question = typed or st.session_state.pop("pending_question", None)

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the schemes document..."):
                result = answer_question(
                    vectorstore, llm, question,
                    history=st.session_state["history"],
                    schemes=selected_schemes or None,
                )
        except Exception as error:  # network, quota or key problems
            st.error(f"Could not get an answer: {error}")
            st.stop()

        st.markdown(f'<div class="answer-card">{result["answer"]}</div>', unsafe_allow_html=True)
        if result["pages"]:
            st.caption("Source pages: " + ", ".join(str(p) for p in result["pages"]))
        if result["docs"] and result["found"]:
            with st.expander("📄 View retrieved chunks"):
                for doc in result["docs"]:
                    st.write(doc.page_content)
                    tags = ", ".join(doc.metadata.get("schemes", [])) or "no scheme tag"
                    st.caption(f"Page {doc.metadata.get('page_number')} · chunk {doc.metadata.get('chunk_id')} · {tags}")
                    st.divider()

    st.session_state["history"].append(
        {"question": question, "answer": result["answer"], "pages": result["pages"]}
    )
