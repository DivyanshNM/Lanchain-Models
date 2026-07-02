from langchain_core.chat_history import InMemoryChatMessageHistory
from chains import chain, router_chain,general_chain

history = InMemoryChatMessageHistory()

print("ChatBot Started (type 'exit' to quit)\n")

#Disclamer
DISCLAIMER = (
    "This information is provided for general educational purposes only "
    "and does not constitute legal advice. Laws depend on the specific "
    "facts of each case. Please consult a qualified advocate for advice "
    "tailored to your situation."
)
#Chat Loop

while True:

    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        print("\nGoodbye!")
        break

    try:
        intent = router_chain.invoke({
            "input": user_input
        }).content.strip().upper()

        if intent=="LEGAL":
            if len(user_input.strip())<5:
                print("please decribe you issue in details")
                continue
            result = chain.invoke({
                "chat_history": history.messages,
                "input": user_input
            })
            result.disclaimer=DISCLAIMER
            history.add_user_message(user_input)
            history.add_ai_message(result.model_dump_json(indent=2))

            print("\n---------- LEGAL ANALYSIS ----------\n")

            print("\nSummary:")
            print(result.summary)

            print("\nFacts:")
            for fact in result.facts:
                print("-", fact)

            print("\nViolation:")
            print(result.possible_violation)

            print("\nMissing Information:")

            if result.missing_information:
                for item in result.missing_information:
                    print("-", item)
            else:
                print("None")

            print("\nRecommended Actions:")

            if result.recommended_actions:
                for action in result.recommended_actions:
                    print("-", action)
            else:
                print("None")

            print("\nAuthorities:")

            if result.authorities:
                for auth in result.authorities:
                    print(f"\nAuthority : {auth.name}")
                    print(f"Reason    : {auth.reason}")
            else:
                print("None")

            print("\nEvidence:")

            if result.evidence:
                for ev in result.evidence:
                    print(f"\nEvidence : {ev.item}")
                    print(f"Reason   : {ev.reason}")
            else:
                print("None")

            print("\nUrgent:")
            print("Yes" if result.urgent else "No")   
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
                    print(f"\nAct                  : {law.law}")
                    print(f"Section              : {law.section}")
                    print(f"Title                : {law.title}")
                    print(f"Purpose              : {law.purpose}")
                    print(f"Why Applicable       : {law.applicability_reason}")
                    print(f"Punishment           : {law.punishment}")
            else:
                print("None")

            print("\nConfidence:")
            print(result.confidence)

            print("\nDisclaimer:")
            print(result.disclaimer)

            print("\n------------------------------------------\n")
        else:
            response = general_chain.invoke({
                "chat_history": history.messages,
                "input": user_input
            })

            history.add_user_message(user_input)
            history.add_ai_message(response.content)

            print("\nAssistant:")
            print(response.content)
    except Exception as e:
        print("\nError:", e)