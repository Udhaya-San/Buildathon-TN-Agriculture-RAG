from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

# Load API Key
load_dotenv()

# Load Embedding Model
embeddings = OpenAIEmbeddings()

# Load FAISS Database
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("✅ FAISS Database Loaded Successfully")

# Create Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Load LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Dynamic User Query
query = input("\nAsk your Agriculture Question: ")

# Retrieve Relevant Chunks
results = retriever.invoke(query)
print("\nSources:")

for doc in results:
    print(f"Page Number: {doc.metadata.get('page')}")

# Build Context
context = ""

for doc in results:
    context += doc.page_content + "\n\n"

prompt = f"""
You are a Tamil Nadu Agriculture Scheme Assistant.

Rules:
1. Answer only from the provided context.
2. If the answer is not present in the context, say:
   "Information not available in the Agriculture Schemes document."
3. Do not make assumptions.
4. Mention important scheme names when available.

Context:
{context}

Question:
{query}

Answer:
"""

# Generate Response
response = llm.invoke(prompt)

# Display Result
print("\n" + "=" * 60)
print("QUESTION")
print("=" * 60)
print(query)

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)
print(response.content)