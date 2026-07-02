from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser
from config import llm, router_llm
from models import LegalResponse
from prompts import router_prompt, get_legal_prompt, general_prompt


parser = PydanticOutputParser(
    pydantic_object=LegalResponse
)

model = ChatHuggingFace(llm=llm)

router_model = ChatHuggingFace(llm=router_llm)

prompt=get_legal_prompt(parser)

chain = prompt | model | parser

router_chain = router_prompt | router_model

general_chain = general_prompt | model