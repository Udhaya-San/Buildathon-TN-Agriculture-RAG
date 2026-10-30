#Since your PDF is 182 pages and entirely in Tamil, we should verify the extraction quality before generating embeddings
#if my pdf contains this kind of letters ����
#□ □ □
#garbled text then it will avoid this
#for this purpose the inspection python file created
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("TN_Agriculture_Schemes.pdf")
documents = loader.load()

# Split PDF into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("=" * 60)
print("FIRST CHUNK")
print("=" * 60)

print(chunks[0].page_content)

print("\n")

print("=" * 60)
print("FIRST CHUNK METADATA")
print("=" * 60)

print(chunks[0].metadata)