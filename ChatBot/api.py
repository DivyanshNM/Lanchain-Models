from fastapi import FastAPI
from pydantic import BaseModel
from history import get_history
from chains import chain,router_chain,general_chain

app = FastAPI()

class ChatRequest(BaseModel):
    conversation_id:str
    prompt: str

@app.post("/chat")
def chat(request: ChatRequest):
    history = get_history(request.conversation_id)
    intent=router_chain.invoke({
        "input":request.prompt
    }).content.strip().upper()
    if intent=="LEGAL":

        result = chain.invoke({
            "chat_history": history.messages,
            "input": request.prompt
        })
        history.add_user_message(request.prompt)
        history.add_ai_message(result.model_dump_json())
        return result.model_dump()
    else:
        result=general_chain.invoke({
            "chat_history":history.messages,
            "input":request.prompt
        })
        history.add_user_message(request.prompt)
        history.add_ai_message(result.model_dump_json())
        return result.model_dump()