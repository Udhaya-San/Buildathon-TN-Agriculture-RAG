from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("TN_Agriculture_Schemes.pdf")
documents = loader.load()

for page_no in [0, 50, 100]:

    print("\n" + "=" * 60)
    print(f"PAGE {page_no}")
    print("=" * 60)

    print(documents[page_no].page_content[:1000])