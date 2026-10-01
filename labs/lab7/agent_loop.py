"""
Lab 7 — สร้าง Agent Loop แรกด้วย Pure Python (ไม่ใช้ framework) [หลักสูตร PDPA]
อ้างอิง outline: บทที่ 1.3 / แบบฝึกหัดที่ 3

หัวใจของ outline: "Minimal Agent = while loop + model + tools"
  - นิยาม local tool 2 ตัว (get_time, calculate) พร้อม schema แบบ OpenAI function
  - เขียน agent loop ที่: ถาม LLM -> ถ้ามี tool_calls ก็เรียก tool แล้ววนกลับ ->
    ถ้าไม่มี tool_calls ถือว่า end_turn -> จบ
  - มี logging แสดง think -> tool call -> observe -> answer

โค้ดนี้คือ "loop ที่เขียนด้วยมือ" ของสิ่งที่หลักสูตรที่ 1 อธิบายเป็นทฤษฎี
(LLM เห็น tools -> ตัดสินใจ tool_use -> client เรียก tool -> ป้อนผลกลับ)

รัน:  python labs/lab7/agent_loop.py "ตอนนี้กี่โมง แล้ว 15*4 เท่ากับเท่าไร"
"""
import sys, os, json, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from labs.core import llm

# ---- (1) Local tools : ฟังก์ชันจริง + schema แบบ OpenAI function ----
def get_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    # ประเมินเฉพาะนิพจน์เลขคณิตอย่างปลอดภัย (ตัวอย่างการเรียนรู้)
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "error: อนุญาตเฉพาะตัวเลขและ + - * / ( )"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"error: {e}"


LOCAL_FUNCS = {"get_time": lambda **_: get_time(),
               "calculate": lambda expression, **_: calculate(expression)}

TOOLS = [
    {"type": "function", "function": {
        "name": "get_time", "description": "คืนวันเวลาปัจจุบัน",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "calculate", "description": "คำนวณนิพจน์เลขคณิต เช่น 15*4",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "นิพจน์ เช่น (15+5)*2"}},
            "required": ["expression"]},
    }},
]

SYSTEM = "คุณเป็นผู้ช่วยที่ใช้ tool ได้ ถ้าจำเป็นให้เรียก tool ก่อนตอบ ตอบเป็นภาษาไทย"


def dispatch(name: str, args: dict) -> str:
    fn = LOCAL_FUNCS.get(name)
    return fn(**args) if fn else f"error: ไม่พบ tool {name}"


# ---- (2) Agent loop : while loop + model + tools ----
def run_agent(question: str, max_steps: int = 6):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    for step in range(1, max_steps + 1):
        resp = llm.chat(messages=messages, tools=TOOLS)
        msg = resp.choices[0].message

        if msg.tool_calls:
            print(f"[step {step}] THINK -> ขอเรียก {len(msg.tool_calls)} tool")
            # ต้อง append assistant message ที่มี tool_calls ก่อน
            messages.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                result = dispatch(call.function.name, args)
                print(f"           TOOL_USE {call.function.name}({args}) -> {result}")
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": result,   # OBSERVE: ป้อนผล tool กลับเข้า context
                })
            continue   # วนกลับให้ LLM อ่านผล tool

        # ไม่มี tool_calls = end_turn
        print(f"[step {step}] END_TURN")
        print("-" * 60)
        print(f"[answer] {msg.content}")
        return msg.content

    print("[!] ถึงขีดจำกัดจำนวนรอบแล้ว")
    return None


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "ตอนนี้กี่โมง แล้ว 15*4 เท่ากับเท่าไร"
    print(f"[user] {q}")
    run_agent(q)
