from pathlib import Path
from langchain_openai import ChatOpenAI

# Singleton variables
_openai_key = None
_llm = None

def get_openai_key():
    global _openai_key
    if _openai_key is None:
        current_path = Path(__file__).resolve()

        for parent in [current_path] + list(current_path.parents):
            env_path = parent / ".env"
            if env_path.exists():
                with env_path.open("r") as f:
                    for line in f:
                        if line.strip().startswith("OPENAI_API_KEY="):
                            _openai_key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                            return _openai_key

        raise ValueError("OPENAI_API_KEY not found in any .env file up the directory tree.")
    return _openai_key

#TODO: Orogat: Add other LLMs, use code from src\chattykg\llms\llm_creation.py
def get_llm(model_name="gpt-4", temperature=0.5):
    """Get a configured LLM instance"""
    global _llm
    if _llm is None:
        # if model from openai, use the openai key
        if model_name.startswith("gpt-"):
            _llm = ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=get_openai_key())
        else:
            _llm = ChatOpenAI(model=model_name, temperature=temperature)
    return _llm

# Default instance for backward compatibility
chatbot_llm = get_llm(model_name="gpt-4o")

