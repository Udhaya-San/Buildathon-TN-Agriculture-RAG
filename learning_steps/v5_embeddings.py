from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings()

query = "நுண்ணீர் பாசன திட்டம்"
print(repr(query))

vector = embeddings.embed_query(query)

print("✅ Embedding Created Successfully")
print(f"Query: {query}")
print(f"Vector Length: {len(vector)}")