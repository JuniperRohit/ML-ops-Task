import os
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel

from .rag import retrieve_context


class SimpleLLM:
    """Simple fallback LLM for offline/testing mode."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.is_offline = api_key is None
    
    def invoke(self, messages):
        """Simulate LLM response."""
        return {"content": "Placeholder answer from offline agent."}


def get_llm():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        return ChatOpenAI(openai_api_key=openai_api_key, temperature=0.2)
    # Local fallback for offline and CI execution
    return SimpleLLM()


@tool
def knowledge_base(query: str) -> str:
    """Retrieve context from the knowledge base."""
    context = retrieve_context(query, top_k=3)
    return context or "No relevant context found."


def create_rohit_react_agent():
    """Create a simple ReAct-like agent for MLOps queries."""
    llm = get_llm()
    
    class SimpleAgent:
        def __init__(self, model):
            self.model = model
            self.tools = [knowledge_base]
        
        def invoke(self, input_dict: Dict[str, str]):
            prompt = input_dict.get("input", "")
            # Try to get context from knowledge base
            try:
                context = knowledge_base(prompt)
                full_prompt = f"{prompt}\n\nContext: {context}"
            except:
                full_prompt = prompt
            
            # Return a simple response object with an output attribute
            class Response:
                def __init__(self, text):
                    self.output = text
            
            return Response(f"Analysis: {full_prompt[:100]}...")
        
        def run(self, prompt: str) -> str:
            """Alias for invoke that works with string prompts."""
            response = self.invoke({"input": prompt})
            return response.output
    
    return SimpleAgent(llm)


class CrewAIOrchestrator:
    def __init__(self):
        self.analyst_agent = create_rohit_react_agent()
        self.explainer_agent = create_rohit_react_agent()

    def run(self, question: str, session_history: Dict[str, str]):
        analyst_prompt = f"You are Analyst. Analyze and summarize key points for: {question}"
        analyst_result = self.analyst_agent.invoke({"input": analyst_prompt}).output

        explainer_prompt = (
            "You are Explainer. Convert Analyst findings into a clear user-friendly answer. "
            f"Analyst notes: {analyst_result}"
        )
        explainer_result = self.explainer_agent.invoke({"input": explainer_prompt}).output

        session_history["last_analyst"] = analyst_result
        session_history["last_explainer"] = explainer_result

        return {"analyst": analyst_result, "explainer": explainer_result}
