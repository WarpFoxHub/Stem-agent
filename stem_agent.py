import os
import json
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

AVAILABLE_TOOLS = ["internet_search", "python_repl", "file_analyzer"]

class AgentState(TypedDict):
    domain: str
    specialization: str
    plan: List[str]
    tools: List[str]  
    critique: str
    result: str

llm = ChatOpenAI(model="gpt-4o")

def stem_cell_logic(state: AgentState):
    print(f"--- STEM: Deep differentiation for {state['domain']} ---")
    prompt = f"""
    You are a meta-agent "Stem Cell". Your DNA allows you to transform into any specialized expert.
    
    DOMAIN: {state['domain']}
    AVAILABLE TOOLS: {AVAILABLE_TOOLS}

    TRANSFORMATION INSTRUCTIONS:
    1. ONTOLOGY DEFINITION: Briefly describe the main challenges in this domain and what skills (hard skills) are needed to solve them.
    2. SYSTEM PROMPT: Formulate a detailed instruction for yourself. Include:
       - Your professional identity.
       - Your communication style.
       - Your constraints (what you should NOT do).
    3. TOOLKIT: List the tools from the above list that you integrate into your work. Justify the choice of each.
    4. ALGORITHM: Describe the step-by-step process of your work (workflow).

    IMPORTANT: Provide the result strictly in JSON format inside a json code block ...
    """
    
    response = llm.invoke(prompt)
    content = response.content
    
    selected_tools = [t for t in AVAILABLE_TOOLS if t in content.lower()]
    
    return {
        "specialization": content,
        "tools": selected_tools,
        "plan": ["Data Search", "Analysis", "Final Report"]
    }

def critic_logic(state: AgentState):
    print("--- CRITIC: Analyzing specialization depth... ---")
    prompt = f"""
    You are an expert in AI agent architecture. Your task: conduct a rigorous audit of the "stem cell" that is trying to specialize in the domain: {state['domain']}.

    Evaluate the provided specialization according to the following criteria:
    1. ROLE DEPTH: Is the System Prompt specific? (Avoid generic phrases like "I will help you").
    2. TOOL SELECTION: Do the selected tools {state['tools']} correspond to domain tasks? Are there any unnecessary or missing tools?
    3. PLAN REALISM: Can this plan be executed in 3 steps, or is it just "filler"?

    If specialization is weak, write 'REJECT' and provide specific remedial instructions.
    If specialization is expert-level, write 'APPROVE'.

    CURRENT SPECIALIZATION:
    {state['specialization']}
    """
    response = llm.invoke(prompt)
    return {"critique": response.content}

def should_continue(state: AgentState):
    return "save_and_execute" if "APPROVE" in state["critique"] else "retry"

import re

def save_json_logic(state: AgentState):
    print("--- SYSTEM: Cleaning and saving configuration ---")
    
    raw_text = state['specialization']
    
    json_match = re.search(r"```json\s+(.*?)\s+```", raw_text, re.DOTALL)
    
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            specialization_clean = data.get("system_prompt", raw_text)
            tools_list = data.get("tools", state['tools'])
        except:
            specialization_clean = raw_text
            tools_list = state['tools']
    else:
        specialization_clean = raw_text
        tools_list = state['tools']

    os.makedirs("registry", exist_ok=True)
    config = {
        "domain": state['domain'],
        "specialization": specialization_clean, 
        "tools": tools_list,
        "plan": state['plan']
    }
    
    filename = f"registry/{state['domain'].replace(' ', '_').lower()}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
        
    return state

def execution_logic(state: AgentState):
    print(f"--- EXECUTOR: Working with tools: {state['tools']} ---")
    prompt = f"As {state['specialization']}, using {state['tools']}, solve the task {state['domain']}"
    response = llm.invoke(prompt)
    return {"result": response.content}

workflow = StateGraph(AgentState)
workflow.add_node("stem_cell", stem_cell_logic)
workflow.add_node("critic", critic_logic)
workflow.add_node("save_json", save_json_logic)
workflow.add_node("specialized_agent", execution_logic)

workflow.set_entry_point("stem_cell")
workflow.add_edge("stem_cell", "critic")
workflow.add_conditional_edges("critic", should_continue, {"retry": "stem_cell", "save_and_execute": "save_json"})
workflow.add_edge("save_json", "specialized_agent")
workflow.add_edge("specialized_agent", END)

app = workflow.compile()

if __name__ == "__main__":
    app.invoke({"domain": "Game Marketing Analyst", "tools": []})