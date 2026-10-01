"""
core — โมดูลกลางที่ใช้ร่วมกันทุก Lab ของหลักสูตร Agentic AI Development with Python (หลักสูตรที่ 2)

แต่ละ Lab จะค่อย ๆ เพิ่มความสามารถ โดยใช้ของกลางจากที่นี่:
  - config.py  : โหลด .env, ค่าตั้งต้น (Lab 1)
  - llm.py     : OpenRouter client (thin client) + helper (Lab 1-2)
  - mcp_client.py : MCP client ผ่าน Streamable HTTP (Lab 4)
  - registry.py   : Tool Registry รวม tools จากหลาย MCP server (Lab 4)

ปรัชญา: "Minimal Agent = while loop + model + tools" — เขียนเองด้วย Pure Python
ก่อน แล้วจึงไปเทียบกับ framework (LangGraph) ใน Lab 8
"""
