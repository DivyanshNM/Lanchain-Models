from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()
token= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=token
)
history=InMemoryChatMessageHistory()
model=ChatHuggingFace(llm=llm)
print("chatBot started (type 'exit' to quit)\n")

while True:
    user_input=input("you: ")
    if user_input.lower()=="exit":
        break
    history.add_user_message(user_input)
    response=model.invoke(history.messages)

    history.add_ai_message(response.content)
    print(f"Bot: {response.content}\n")