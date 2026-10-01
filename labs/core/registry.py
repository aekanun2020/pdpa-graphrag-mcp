"""core.registry — Tool Registry รวม tools จากหลาย MCP server (เริ่มใช้ Lab 4)

หน้าที่:
  1) เชื่อม MCP server หลายตัว (initialize + tools/list)
  2) แปลง MCP inputSchema -> OpenAI function schema (ใช้ตรงได้เพราะเป็น JSON Schema)
  3) map tool_name -> client ตัวที่เป็นเจ้าของ เพื่อ dispatch ตอน LLM ขอเรียก tool

โครงสร้างนี้ "รองรับหลาย MCP server" แต่ในหลักสูตรนี้เราต่อ MCP MSSQL จริงตัวเดียว
เป็นแกน (สอดคล้องกับ Lab 8) — เพิ่ม server อื่น (เช่น RAG :8000) ได้โดยเรียก
add_server() เพิ่มอีกตัวโดยไม่ต้องแก้ agent loop
"""
from .mcp_client import MCPClient


def mcp_to_openai_tools(mcp_tools: list[dict]) -> list[dict]:
    """แปลงรายการ MCP tools -> รูปแบบ tools ของ OpenAI function calling."""
    out = []
    for t in mcp_tools:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                # MCP inputSchema คือ JSON Schema = parameters ของ OpenAI ได้เลย
                "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
            },
        })
    return out


class ToolRegistry:
    """รวม tools จากหลาย MCP server ไว้ในที่เดียว แล้ว dispatch ให้ถูกตัว."""

    def __init__(self):
        self._clients: list[MCPClient] = []
        self._tool_owner: dict[str, MCPClient] = {}   # tool_name -> client
        self._openai_tools: list[dict] = []

    def add_server(self, url: str) -> int:
        """เชื่อม MCP server 1 ตัว, ค้นพบ tools, รวมเข้า registry. คืนจำนวน tools ที่เพิ่ม."""
        client = MCPClient(url)
        client.initialize()
        tools = client.list_tools()
        for t in tools:
            self._tool_owner[t["name"]] = client
        self._openai_tools.extend(mcp_to_openai_tools(tools))
        self._clients.append(client)
        return len(tools)

    @property
    def openai_tools(self) -> list[dict]:
        """tools ทั้งหมดในรูปแบบ OpenAI — เอาไปใส่ใน chat.completions ได้ตรง ๆ."""
        return self._openai_tools

    @property
    def tool_names(self) -> list[str]:
        return list(self._tool_owner.keys())

    def dispatch(self, name: str, arguments: dict) -> str:
        """เรียก tool ตามชื่อ ไปยัง MCP server เจ้าของที่ถูกต้อง."""
        client = self._tool_owner.get(name)
        if client is None:
            return f"[registry] ไม่พบ tool ชื่อ '{name}'"
        return client.call_tool(name, arguments)

    def close(self):
        for c in self._clients:
            c.close()
