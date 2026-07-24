from langchain_chroma import Chroma

from loaders import load_documents
from splitter import split_documents
from embeddings import get_embedding_model


def create_vector_store():
    # Load documents
    documents = load_documents()

    # Split into chunks
    chunks = split_documents(documents)

    # Load embedding model
    embeddings = get_embedding_model()

    # Create Chroma database
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore

if __name__ == "__main__":
    vectorstore = create_vector_store()
    print("Chunks stored:", vectorstore._collection.count())