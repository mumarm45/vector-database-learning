# Initialize Anthropic client
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic

load_dotenv()

def llm_model_langchain(params=None):

    load_dotenv()
    params = params or {}

    api_key = params.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Please set the ANTHROPIC_API_KEY in your .env file")

    model = params.get("model", "claude-3-haiku-20240307")
    max_tokens = params.get("max_tokens", 400)
    temperature = params.get("temperature", 0.7)


    llm = ChatAnthropic(
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return llm


def get_llm_model():
    """Get the LLM model"""
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    MODEL = os.getenv("LLM_MODEL", "claude-3-haiku-20240307")
    return client


def generate_response(prompt: str) -> str:
    """Generate response using LLM model"""
    client = get_llm_model()
    message = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text