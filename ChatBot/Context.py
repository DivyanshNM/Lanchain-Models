from retriever import get_retriever

def get_context(query:str)->str:
    retriever = get_retriever()

    documents = retriever.invoke(query)
    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in documents
    )
    return context