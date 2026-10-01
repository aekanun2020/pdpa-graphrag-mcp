"""core.mcp_client — MCP client ขั้นต่ำผ่าน Streamable HTTP (เริ่มใช้ Lab 4)

ทำ MCP lifecycle ด้วยมือ (ไม่พึ่ง framework) เพื่อให้เห็นว่า "ใต้ฝา" ของ MCP client
ในหลักสูตรที่ 1 (Claude Desktop / LangFlow) ทำงานอย่างไร:

    initialize  ->  notifications/initialized  ->  tools/list  ->  tools/call

MCP Streamable HTTP ตอบกลับเป็น Server-Sent Events (SSE) — โค้ดนี้แกะ event
`data:` ออกมาเป็น JSON ให้เอง รองรับการคงค่า session id ข้าม request
"""
import json
import httpx


class MCPClient:
    """MCP client ขั้นต่ำสำหรับ 1 server (Streamable HTTP)."""

    def __init__(self, url: str, timeout: float = 60.0):
        self.url = url
        self._client = httpx.Client(timeout=timeout)
        self._session_id: str | None = None
        self._id = 0

    # ---- ภายใน: ส่ง JSON-RPC แล้วแกะผลจาก SSE/JSON ----
    def _headers(self) -> dict:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            h["mcp-session-id"] = self._session_id
        return h

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    @staticmethod
    def _parse_sse(text: str):
        """ดึง JSON ก้อนแรกจาก payload ที่อาจเป็น SSE (`data: {...}`) หรือ JSON ตรง ๆ."""
        text = text.strip()
        if not text:
            return None
        if text.startswith("{"):
            return json.loads(text)
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
                if data and data != "[DONE]":
                    return json.loads(data)
        return None

    def _rpc(self, method: str, params: dict | None = None, notify: bool = False):
        body = {"jsonrpc": "2.0", "method": method}
        if not notify:
            body["id"] = self._next_id()
        if params is not None:
            body["params"] = params

        resp = self._client.post(self.url, headers=self._headers(), json=body)
        resp.raise_for_status()

        # เก็บ session id ที่ server ออกให้ (ใช้ซ้ำในทุก request ถัดไป)
        sid = resp.headers.get("mcp-session-id")
        if sid:
            self._session_id = sid

        if notify:
            return None
        return self._parse_sse(resp.text)

    # ---- MCP lifecycle ----
    def initialize(self) -> dict:
        result = self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "course2-pure-python-agent", "version": "1.0"},
        })
        # แจ้ง server ว่า init เสร็จ (ตาม lifecycle ของ MCP)
        self._rpc("notifications/initialized", notify=True)
        return result

    def list_tools(self) -> list[dict]:
        result = self._rpc("tools/list")
        return (result or {}).get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        content = (result or {}).get("result", {}).get("content", [])
        # รวม text ทุกก้อนเป็นสตริงเดียว (พอสำหรับการป้อนกลับให้ LLM)
        parts = [c.get("text", "") for c in content if c.get("type") == "text"]
        return "\n".join(parts) if parts else json.dumps(result, ensure_ascii=False)

    def close(self):
        self._client.close()
