from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()

def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    messages = [
        HumanMessage("""
You are a dataset generator. Generate 20 unique natural-sounding English sentences. Each sentence should contain one or more mountain names drawn from a diverse set of global mountains (e.g., Mount Everest, K2, Denali, Aconcagua, Kilimanjaro, Mont Blanc, Mt. Fuji, Table Mountain). Return JSON array; each item must be:

{
  "text": "<full sentence>",
  "entities": [
    {"start": <start_char>, "end": <end_char>, "label": "MOUNTAIN"}
  ]
}

Requirements:
- Use diverse forms: "Mount X", "Mt. X", "X", "Monte X", local variants where appropriate.
- Include examples with punctuation, parentheses, list contexts ("... and ..."), abbreviations ("Mt."), and possessive ("K2's prominence").
- Include negative examples: sentences with words like "mountain" used metaphorically or place names that are not mountains (mark entities only when they are actual mountains).
- No other entity labels (only MOUNTAIN).
""")
    ]
    response = llm.invoke(messages)

    print(response.content[8:-3])

if __name__ == "__main__":
    main()