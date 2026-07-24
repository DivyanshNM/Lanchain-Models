from pydantic import BaseModel, Field
from typing import List

class ComplaintLaw(BaseModel):
    law: str = Field(description="Name of the Act")
    section: str = Field(description="Section number")
    reason: str = Field(description="Why this section applies")


class ApplicableLaw(BaseModel):
    law: str = Field(description="Name of the Act.")
    section: str = Field(description="Section number.")
    title: str = Field(description="Official title or name of the legal provision.")
    purpose: str = Field(description="Purpose of this legal provision.")
    applicability_reason: str = Field(
        description="Explain why this law applies to the user's situation using only the facts provided."
    )
    punishment: str = Field(
        description="Punishment prescribed under this section. If unknown, write 'Not specified'."
    )


class Authority(BaseModel):
    name: str = Field(description="Authority or department to contact")
    reason: str = Field(description="Why this authority is relevant")


class Evidence(BaseModel):
    item: str = Field(description="Evidence to collect")
    reason: str = Field(description="Why it is important")


class LegalResponse(BaseModel):

    summary: str = Field(
        description="Brief summary of the user's situation."
    )

    facts: List[str] = Field(
        description="Important facts extracted from the user's statement."
    )

    possible_violation: str = Field(
        description="Possible legal violation. Write 'None' if no violation exists."
    )

    complaint_laws: List[ComplaintLaw] = Field(
        description="Laws under which the user may file a complaint."
    )

    applicable_laws: List[ApplicableLaw] = Field(
        description="Relevant legal provisions applicable to the case."
    )

    missing_information: List[str] = Field(
        description="Important information required before giving a better legal analysis."
    )

    recommended_actions: List[str] = Field(
        description="Suggested next steps for the user."
    )

    authorities: List[Authority] = Field(
        description="Authorities or government departments the user should approach."
    )

    evidence: List[Evidence] = Field(
        description="Evidence the user should preserve or collect."
    )

    urgent: bool = Field(
        description="True if immediate action is required."
    )

    confidence: str = Field(
        description="High, Medium or Low."
    )

    disclaimer: str = Field(
        description="General legal disclaimer."
    )