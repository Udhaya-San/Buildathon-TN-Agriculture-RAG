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