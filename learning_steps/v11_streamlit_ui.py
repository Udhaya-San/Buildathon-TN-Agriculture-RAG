import streamlit as st

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

# ----------------------------------
# Load Environment
# ----------------------------------

load_dotenv()

# ----------------------------------
# Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="TN Agriculture Schemes Assistant",
    page_icon="🌾",
    layout="wide"
)

# ----------------------------------
# Custom CSS
# ----------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #f6f8f5;
}

.gov-header {
    background-color: #1b5e20;
    color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

.main-title {
    color: #0d5c1d;
    font-size: 42px;
    text-align: center;
    font-weight: bold;
}

.sub-title {
    color: #333333;
    text-align: center;
    font-size: 18px;
}

.answer-card {
    background-color: white;
    color: black;
    padding: 20px;
    border-radius: 10px;
    border-left: 8px solid #2e7d32;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.15);
}

.stButton > button {
    background-color: #2e7d32;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #1b5e20;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# Government Header
# ----------------------------------

st.markdown("""
<div class="gov-header">
    <h2>Government of Tamil Nadu</h2>
    <h3>Department of Agriculture & Farmers Welfare</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------
# Title
# ----------------------------------

st.markdown(
    """
    <div class="main-title">
        🌾 Tamil Nadu Agriculture Schemes Assistant
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
        Search Agriculture Schemes, Subsidies, Crop Insurance,
        Organic Farming and Farmer Welfare Programs
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------
# Banner Image
# ----------------------------------

try:
    st.image(
        "assets/agriculture_banner.jpg",
        use_container_width=True
    )
except:
    pass

# ----------------------------------
# Dashboard Metrics
# ----------------------------------

st.markdown("## 📊 Knowledge Base")

col1, col2, col3 = st.columns(3)

col1.metric("📄 Pages", "181")
col2.metric("📚 Chunks", "1100")
col3.metric("⚡ Vector DB", "FAISS")

st.divider()

# ----------------------------------
# Sidebar
# ----------------------------------

with st.sidebar:

    try:
        st.image(
            "assets/tn_logo.png",
            width=120
        )
    except:
        pass

    st.title("🌾 Agriculture Portal")

    st.success("PDF Loaded")

    st.success("FAISS Active")

    st.success("GPT-4o-mini Active")

    st.markdown("---")

    st.markdown("""
### About

This Assistant helps users search:

✅ Agriculture Schemes

✅ Crop Insurance

✅ Organic Farming

✅ Irrigation Schemes

✅ Farmer Welfare Programs
""")

# ----------------------------------
# Load Components
# ----------------------------------

@st.cache_resource
def load_components():

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    return retriever, llm

retriever, llm = load_components()

# ----------------------------------
# Popular Searches
# ----------------------------------

st.markdown("## 🔍 Popular Searches")

c1, c2, c3, c4 = st.columns(4)

if c1.button("🌱 Organic Farming"):
    st.session_state["query"] = "Organic Farming"

if c2.button("💧 Irrigation"):
    st.session_state["query"] = "Irrigation Schemes"

if c3.button("🌾 Crop Insurance"):
    st.session_state["query"] = "Crop Insurance"

if c4.button("🏪 e-NAM"):
    st.session_state["query"] = "e-NAM"

# ----------------------------------
# Search Box
# ----------------------------------

query = st.text_area(
    "Ask Your Agriculture Question",
    value=st.session_state.get("query", ""),
    placeholder="""
Examples:

What are Organic Farming Schemes?

What is Crop Insurance?

What is e-NAM?

நுண்ணீர் பாசன திட்டம் என்றால் என்ன?
""",
    height=120
)

# ----------------------------------
# Search Button
# ----------------------------------

if st.button("🔎 Search Agriculture Schemes"):

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching Agriculture Schemes..."):

            results = retriever.invoke(query)

            context = ""
            source_pages = []

            for doc in results:

                context += doc.page_content + "\n\n"

                page = doc.metadata.get("page")

                if page not in source_pages:
                    source_pages.append(page)

            prompt = f"""
You are a Tamil Nadu Agriculture Schemes Assistant.

Rules:
1. Answer only from the given context.
2. Do not make assumptions.
3. If information is unavailable, reply:
Information not available in the Agriculture Schemes document.

Context:
{context}

Question:
{query}

Answer:
"""

            response = llm.invoke(prompt)

        st.divider()

        st.subheader("📌 AI Answer")

        st.markdown(
            f"""
            <div class="answer-card">
            {response.content}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        st.subheader("📖 Source Pages")

        cols = st.columns(len(source_pages))

        for i, page in enumerate(source_pages):

            cols[i].metric(
                label="Page",
                value=page
            )

        st.divider()
        with st.expander("📄 View Retrieved Chunks"):

            for index, doc in enumerate(results, start=1):

                st.markdown(f"### Chunk {index}")

                st.write(doc.page_content)

                st.caption(
                    f"Source Page : {doc.metadata.get('page')}"
                )

                st.divider()