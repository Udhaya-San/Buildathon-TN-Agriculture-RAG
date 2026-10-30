from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("TN_Agriculture_Schemes.pdf")
documents = loader.load()

print(f"✅ PDF loaded successfully.")
print(f"✅ Total pages available in the PDF: {len(documents)}")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"✅ Total chunks created: {len(chunks)}")