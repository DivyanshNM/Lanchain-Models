from fastapi import FastAPI
from pydantic import BaseModel
from history import get_history
from chains import chain

app = FastAPI()

class ChatRequest(BaseModel):
    conversation_id:str
    prompt: str

@app.post("/chat")
def chat(request: ChatRequest):
    history = get_history(request.conversation_id)
    result = chain.invoke({
        "chat_history": history.messages,
        "input": request.prompt
    })
    history.add_user_message(request.prompt)
    history.add_ai_message(result.model_dump_json())
    return result.model_dump()