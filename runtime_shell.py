import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

def run_shell():
    print("--- AVAILABLE AGENTS IN REGISTRY ---")
    files = os.listdir("registry")
    for i, f in enumerate(files):
        print(f"{i}. {f}")
    
    choice = int(input("Select agent number: "))
    with open(f"registry/{files[choice]}", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    print(f"\nAgent {config['domain']} is ready to work!")
    
    search = DuckDuckGoSearchRun()
    llm = ChatOpenAI(model="gpt-4o")
    
    while True:
        user_input = input("\nYou (type 'exit' to quit): ")
        if user_input.lower() == 'exit': break
        
        context = ""
        if "internet_search" in config["tools"]:
            print("[System: Agent is using search...]")
            context = search.run(user_input)
            
        full_prompt = f"INSTRUCTION: {config['specialization']}\nCONTEXT FROM NETWORK: {context}\nQUESTION: {user_input}"
        response = llm.invoke(full_prompt)
        print(f"\nAGENT: {response.content}")

if __name__ == "__main__":
    run_shell()