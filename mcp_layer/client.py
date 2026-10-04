"""
mcp_layer/client.py
Discovers and calls tools across all connected MCP servers.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from core.config import settings

SERVERS_DIR = os.path.join(os.path.dirname(__file__), "servers")


class MCPToolClient:
    def __init__(self, server_names: list[str] | None = None):
        self.server_names = server_names or settings.mcp_enabled_servers
        self._stack = AsyncExitStack()
        self._tool_to_session: dict[str, ClientSession] = {}
        self._tools_cache: list[Any] = []

    async def connect(self) -> None:
        for server_name in self.server_names:
            script_path = os.path.join(SERVERS_DIR, f"{server_name}.py")
            if not os.path.exists(script_path) or os.path.getsize(script_path) == 0:
                print(
                    f"[MCP] Skipping '{server_name}': file missing or empty.",
                    file=sys.stderr,
                )
                continue
            try:
                await self._connect_one(server_name, script_path)
            except Exception as exc:
                print(
                    f"[MCP] Failed to connect to '{server_name}': {exc}",
                    file=sys.stderr,
                )

    async def _connect_one(self, server_name: str, script_path: str) -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=[script_path],
        )
        stdio, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()

        tools_resp = await session.list_tools()
        for tool in tools_resp.tools:
            self._tool_to_session[tool.name] = session
            self._tools_cache.append(tool)

        print(
            f"[MCP] '{server_name}' connected, "
            f"tools: {[tool.name for tool in tools_resp.tools]}",
            file=sys.stderr,
        )

    def list_tools(self):
        return list(self._tools_cache)

    async def call_tool(self, name: str, arguments: dict) -> str:
        session = self._tool_to_session.get(name)
        if session is None:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Tool '{name}' is not available.",
                },
                ensure_ascii=False,
            )

        result = await session.call_tool(name, arguments)

        # Newer MCP implementations may expose structuredContent. Prefer it so
        # callers receive valid JSON rather than reparsing a human string.
        structured = getattr(result, "structuredContent", None)
        if structured is None:
            structured = getattr(result, "structured_content", None)
        if structured is not None:
            try:
                return json.dumps(structured, ensure_ascii=False, default=str)
            except Exception:
                pass

        text_parts: list[str] = []
        for content in result.content:
            if hasattr(content, "text"):
                text = str(content.text or "").strip()
                if text:
                    text_parts.append(text)

        return "\n".join(text_parts) if text_parts else ""

    async def close(self) -> None:
        await self._stack.aclose()
