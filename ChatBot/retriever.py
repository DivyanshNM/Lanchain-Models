from langchain_chroma import Chroma

from embeddings import get_embedding_model

embeddings = get_embedding_model()

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

def get_retriever():

    return retriever