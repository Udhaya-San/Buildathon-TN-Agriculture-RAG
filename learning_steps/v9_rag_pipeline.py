from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()

# Load Embedding Model
embeddings = OpenAIEmbeddings()

# Load Vector DB
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Load LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# User Query
query = "What are the benefits of Organic Farming Scheme?"

# Retrieve Relevant Chunks
results = retriever.invoke(query)

# Create Context
context = ""

for doc in results:
    context += doc.page_content + "\n\n"

# Prompt
prompt = f"""
You are a Tamil Nadu Agriculture Scheme Assistant.

Answer ONLY from the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

# Generate Answer
response = llm.invoke(prompt)

print("\nQUESTION:")
print(query)

print("\nANSWER:")
print(response.content)