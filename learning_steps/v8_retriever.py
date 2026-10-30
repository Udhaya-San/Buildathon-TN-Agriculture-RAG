from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

print("Loading Embedding Model...")

embeddings = OpenAIEmbeddings()

print("Loading FAISS Database...")

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

print("✅ Retriever Created Successfully")

# Query
query = "Organic Farming"

print(f"\nSearching for: {query}")

# Retrieve Documents
results = retriever.invoke(query)

print("\n✅ Top 3 Matching Results Found\n")

for i, doc in enumerate(results, start=1):

    print("=" * 60)
    print(f"RESULT {i}")
    print("=" * 60)

    print(doc.page_content[:500])

    print("\nPage Number:", doc.metadata.get("page"))

    print()