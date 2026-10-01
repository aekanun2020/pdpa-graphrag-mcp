#!/usr/bin/env bash
# Talk to the MCP server with plain curl. Needs jq.
# A bare request without the Accept header returns -32600 "Client must accept text/event-stream":
# that means the server is up; streamable-http just requires the header below.
URL=${MCP_SERVER_URL:-http://localhost:8100/mcp}
H=(-H "Content-Type: application/json" -H "Accept: application/json, text/event-stream")

echo "== tools/list =="
curl -s -X POST "$URL" "${H[@]}" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | jq '.result.tools[].name'

echo "== get_penalty(sec_26) =="
curl -s -X POST "$URL" "${H[@]}" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_penalty","arguments":{"section_id":"sec_26"}}}' \
  | jq '.result.content[0].text | fromjson'

echo "== search_pdpa =="
curl -s -X POST "$URL" "${H[@]}" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search_pdpa","arguments":{"query":"เก็บข้อมูลสุขภาพพนักงานได้ไหม","top_k":5}}}' \
  | jq '.result.content[0].text | fromjson | .keywords'
