import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

def evaluate():
    domain = "Cryptocurrency Market Analyst"
    query = "Analyze the current Bitcoin situation and provide a forecast for the week."
    
    llm = ChatOpenAI(model="gpt-4o")
    
    res_normal = llm.invoke(query)
    
    filename = f"registry/{domain.replace(' ', '_').lower()}.json"
    with open(filename, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    res_stem = llm.invoke([
        {"role": "system", "content": config["specialization"]},
        {"role": "user", "content": query}
    ])
    
    print("\n" + "="*50)
    print(f"QUERY: {query}")
    print("="*50)
    
    print("\n--- NORMAL MODEL RESPONSE ---")
    print(res_normal.content)
    
    print("\n" + "-"*30)
    
    print("\n--- STEM-AGENT RESPONSE ---")
    print(res_stem.content)
    
    print("\n" + "="*50)
    print("=== METRICS ===")
    print(f"Length (Normal): {len(res_normal.content)}")
    print(f"Length (Stem):   {len(res_stem.content)}")

if __name__ == "__main__":
    evaluate()