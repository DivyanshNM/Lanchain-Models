from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
prompt=ChatPromptTemplate.from_messages([
    ("system","You are a helpful and experienced Indian Advocate."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{input}")
])
chain =prompt|model
print("chatBot started (type 'exit' to quit)\n")

while True:
    user_input=input("you: ")
    if user_input.lower()=="exit":
        break
    response=chain.invoke({
        "chat_history":history.messages,
        "input":user_input
    })

    history.add_user_message(user_input)
    history.add_ai_message(response.content)
    print(f"Bot: {response.content}\n")