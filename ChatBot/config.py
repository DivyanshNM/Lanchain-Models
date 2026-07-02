from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load Environment Variables

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Main LLM

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=token,
    temperature=0.1,
    max_new_tokens=1200,
)

# Router LLM

router_llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=token,
    temperature=0,
    max_new_tokens=10,
)