import os
from typing import Any

from tavily import TavilyClient


class WebSearchTool:
    """Search the web for information using the Tavily API."""

    def __init__(self, api_key: str | None = None, max_results: int = 5) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("TAVILY_API_KEY")
        self.max_results = max_results

    def __call__(self, query: str) -> dict[str, Any]:
        if not self.api_key:
            return {"ok": False, "error": "missing TAVILY_API_KEY"}

        client = TavilyClient(api_key=self.api_key)
        try:
            response = client.search(query, max_results=self.max_results, include_answer=True)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        results = [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "content": str(item.get("content", "")),
                "score": item.get("score", 0.0),
            }
            for item in response.get("results", []) or []
        ]

        return {
            "ok": True,
            "query": query,
            "answer": str(response.get("answer") or ""),
            "results": results,
        }
