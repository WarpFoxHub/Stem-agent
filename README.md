# Stem Cell AI: Self-Differentiating Agent Factory
This project implements a Self-Differentiating AI Agent based on a biological stem cell metaphor. It allows a "Stem" model to autonomously research a domain, define its own professional persona, and select required tools to solve complex tasks without human intervention.

# 🚀 Quick Start
1. Requirements
Python 3.10+

OpenAI API Key (GPT-4o recommended)

Create a .env file in the root directory:

# Running the System
Generate an Agent: Run python stem_agent.py. This will create a specialized JSON configuration in the registry/ folder.

Talk to the Agent: Run python runtime_shell.py. Choose your new agent and start a conversation.

Benchmark: Run python evaluator.py to compare the specialized agent against a base GPT-4o model.

# Project Structure
stem_agent.py — The "Factory" logic using LangGraph.

runtime_shell.py — Interactive shell for agent deployment.

evaluator.py — Comparison tool for performance metrics.

registry/ — Storage for specialized agent configurations (JSON).
