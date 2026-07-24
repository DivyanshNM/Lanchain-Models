from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import (RunnableLambda, RunnablePassthrough)
from config import llm, router_llm
from Context import get_context
from models import LegalResponse
from prompts import router_prompt, get_legal_prompt, general_prompt


parser = PydanticOutputParser(
    pydantic_object=LegalResponse
)

model = ChatHuggingFace(llm=llm)

router_model = ChatHuggingFace(llm=router_llm)

prompt=get_legal_prompt(parser)

chain = (
    {
        "input": RunnableLambda(
            lambda x: x["input"]
        ),
        "chat_history": RunnableLambda(
            lambda x: x["chat_history"]
        ),
        "context": RunnableLambda(
            lambda x: get_context(x["input"])
        )
    }
    | prompt
    | model
    | parser
)

router_chain = router_prompt | router_model

general_chain = general_prompt | model