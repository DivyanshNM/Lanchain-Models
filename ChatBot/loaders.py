from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

def load_documents():
    loader = DirectoryLoader(
        path="ChatBot/data",
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents
