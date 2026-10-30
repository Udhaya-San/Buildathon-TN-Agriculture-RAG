from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("TN_Agriculture_Schemes.pdf")

documents = loader.load()

documents = documents[1:]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("\nCHUNK 100")
print(chunks[100].page_content[:500])

print("\nCHUNK 500")
print(chunks[500].page_content[:500])

print("\nCHUNK 1000")
print(chunks[1000].page_content[:500])