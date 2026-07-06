import os
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

ProviderName = Literal["openai", "deepseek"]


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is not set. Add it to .env or export it in your shell.")
    return value


def create_model(provider: ProviderName = "openai", model: str | None = None) -> ChatOpenAI:
    """Create an OpenAI-compatible chat model after loading environment variables."""

    load_dotenv()

    if provider == "openai":
        return ChatOpenAI(
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=_required_env("OPENAI_API_KEY"),
        )

    if provider == "deepseek":
        return ChatOpenAI(
            model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            api_key=_required_env("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

    raise ValueError(f"Unsupported provider: {provider}")
