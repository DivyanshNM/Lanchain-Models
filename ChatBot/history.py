from langchain_core.chat_history import InMemoryChatMessageHistory

history_store = {}

def get_history(conversation_id: str):

    if conversation_id not in history_store:
        history_store[conversation_id] = InMemoryChatMessageHistory()

    return history_store[conversation_id]