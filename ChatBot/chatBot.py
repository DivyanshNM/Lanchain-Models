from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import os

#Load Environment Variables

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

#LLM 

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=token,
)

#Pydantic Models

class ComplaintLaw(BaseModel):
    law: str = Field(description="Name of the Act")
    section: str = Field(description="Section number")
    reason: str = Field(description="Why this section applies")


class ApplicableLaw(BaseModel):
    law: str = Field(description="Name of the Act")
    section: str = Field(description="Section number")
    purpose: str = Field(description="Purpose of this section")
    punishment: str = Field(description="Punishment under this section")


class LegalResponse(BaseModel):
    summary: str = Field(description="Short summary of the user's problem")

    possible_violation: str = Field(
        description="Possible legal offence. Write 'None' if no offence exists."
    )

    complaint_laws: List[ComplaintLaw]

    applicable_laws: List[ApplicableLaw]

    confidence: str = Field(
        description="High, Medium or Low"
    )

    disclaimer: str = Field(
        description="General legal disclaimer"
    )

# Output Parser 

parser = PydanticOutputParser(pydantic_object=LegalResponse)


model = ChatHuggingFace(llm=llm)

#Chat History 

history = InMemoryChatMessageHistory()

#Prompt

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
    You are a highly experienced Indian Advocate.

    Rules:
    - Use very simple English.
    - Avoid difficult legal vocabulary.
    - Never invent Acts or Section numbers.
    - If unsure, say so.
    - Only provide general legal information.
    - Do not provide personal legal advice.

    Return ONLY valid JSON.

    Rules:
    - Use double quotes only.
    - Never use single quotes.
    - Never wrap the JSON in markdown.
    - Never write explanations.
    - Never write text before or after the JSON.
    - Every applicable_laws object MUST contain:
        - law
        - section
        - purpose
        - punishment
    For every user query return:
    1. Summary of the problem.
    2. Possible legal violation.
    3. Relevant laws under which a complaint may be lodged.
    4. Applicable laws and their punishment.
    5. Confidence level.
    6. Disclaimer.

    {format_instructions}
    """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]).partial(
        format_instructions=parser.get_format_instructions()
    )

#Chain

chain = prompt | model | parser

print("ChatBot Started (type 'exit' to quit)\n")

#Chat Loop

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("\nGoodbye!")
        break

    try:

        result = chain.invoke({
            "chat_history": history.messages,
            "input": user_input
        })

        history.add_user_message(user_input)
        history.add_ai_message(str(result))

        print("\n---------- LEGAL ANALYSIS ----------\n")

        print("Summary:")
        print(result.summary)

        print("\nPossible Violation:")
        print(result.possible_violation)

        print("\nComplaint Laws:")

        if result.complaint_laws:
            for law in result.complaint_laws:
                print(f"\nAct      : {law.law}")
                print(f"Section  : {law.section}")
                print(f"Reason   : {law.reason}")
        else:
            print("None")

        print("\nApplicable Laws:")

        if result.applicable_laws:
            for law in result.applicable_laws:
                print(f"\nAct         : {law.law}")
                print(f"Section     : {law.section}")
                print(f"Purpose     : {law.purpose}")
                print(f"Punishment  : {law.punishment}")
        else:
            print("None")

        print("\nConfidence:")
        print(result.confidence)

        print("\nDisclaimer:")
        print(result.disclaimer)

        print("\n------------------------------------------\n")

    except Exception as e:
        print("\nError:", e)