"""
mcp_layer/servers/web_search_server.py

MCP server exposing one public-web tool backed by Tavily.
Internal RCRC facts must never be fetched through this tool.
"""

from __future__ import annotations

import os
import sys
from typing import Annotated

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    ),
)

import requests
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from core.config import settings

mcp = FastMCP("web-search-server")

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


@mcp.tool()
def web_search(
    query: Annotated[
        str,
        Field(
            description=(
                "Concise public web-search query. Use only for external/public "
                "information, never internal RCRC employee/system facts."
            )
        ),
    ],
    topic: Annotated[
        str,
        Field(description="Search topic: general or news"),
    ] = "general",
    max_results: Annotated[
        int,
        Field(description="Maximum number of results, from 1 to 10"),
    ] = 5,
    search_depth: Annotated[
        str,
        Field(description="Search depth: basic or advanced"),
    ] = "basic",
    days: Annotated[
        int | None,
        Field(
            description=(
                "Optional recency window in days for time-sensitive searches."
            )
        ),
    ] = None,
) -> dict:
    """Search the public internet and return clean, source-backed results."""

    cleaned_query = str(query or "").strip()
    if not cleaned_query:
        return {
            "success": False,
            "error": "No search query provided.",
            "results": [],
        }

    if not settings.enable_web_search:
        return {
            "success": False,
            "error": "Web search is disabled.",
            "results": [],
        }

    api_key = str(settings.tavily_api_key or "").strip()
    if not api_key:
        return {
            "success": False,
            "error": "Tavily API key is missing.",
            "results": [],
        }

    normalized_topic = str(topic or "general").strip().lower()
    if normalized_topic not in {"general", "news"}:
        normalized_topic = "general"

    normalized_depth = str(search_depth or "basic").strip().lower()
    if normalized_depth not in {"basic", "advanced"}:
        normalized_depth = "basic"

    try:
        result_limit = max(1, min(int(max_results), 10))
    except (TypeError, ValueError):
        result_limit = 5

    payload: dict[str, object] = {
        "query": cleaned_query,
        "topic": normalized_topic,
        "search_depth": normalized_depth,
        "max_results": result_limit,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }

    if days is not None:
        try:
            payload["days"] = max(1, min(int(days), 365))
        except (TypeError, ValueError):
            pass

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        response = requests.post(
            TAVILY_SEARCH_URL,
            headers=headers,
            json=payload,
            timeout=settings.tavily_timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Web search timed out.",
            "results": [],
        }
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        return {
            "success": False,
            "error": f"Tavily returned HTTP {status or 'error'}.",
            "results": [],
        }
    except requests.exceptions.RequestException as exc:
        return {
            "success": False,
            "error": f"Could not connect to Tavily: {exc}",
            "results": [],
        }
    except ValueError:
        return {
            "success": False,
            "error": "Tavily returned invalid JSON.",
            "results": [],
        }

    raw_results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(raw_results, list):
        raw_results = []

    cleaned_results: list[dict[str, object]] = []
    for item in raw_results[:result_limit]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        content = str(item.get("content") or "").strip()
        url = str(item.get("url") or "").strip()
        if not (title or content or url):
            continue
        cleaned_results.append(
            {
                "title": title,
                "content": content[:4000],
                "url": url,
                "score": item.get("score"),
                "published_date": item.get("published_date"),
            }
        )

    return {
        "success": True,
        "query": cleaned_query,
        "topic": normalized_topic,
        "results": cleaned_results,
    }


if __name__ == "__main__":
    mcp.run()
