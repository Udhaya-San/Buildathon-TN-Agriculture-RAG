from langchain_community.document_loaders import PyPDFLoader

# Load PDF
loader = PyPDFLoader("TN_Agriculture_Schemes.pdf")

# Read all pages
documents = loader.load()

# Display details
print("✅ PDF loaded successfully.")
print(f"✅ Total pages available in the PDF: {len(documents)}")