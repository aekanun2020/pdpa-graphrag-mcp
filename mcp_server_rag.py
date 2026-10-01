"""
MCP Server for PDPA RAG — FastMCP + streamable-http
====================================================
4 tools: search_pdpa, get_related_sections, get_penalty, get_summary
"""

import os
import json
from mcp.server.fastmcp import FastMCP
from lightrag.main import PDPALightRAG

# ─── Config ────────────────────────────────────────────────

PORT = int(os.environ.get("PORT", "8100"))
HOST = os.environ.get("HOST", "0.0.0.0")
USE_LLM_EXTRACT = os.environ.get("USE_LLM_EXTRACT", "false").lower() == "true"

# ─── Build PDPA LightRAG ตอน startup ─────────────────────

print("🔧 MCP Server PDPA RAG — กำลังเตรียมระบบ...")
rag = PDPALightRAG()
rag.build(use_llm_extract=USE_LLM_EXTRACT, use_llm_profile=False)

# ─── Create FastMCP Server ─────────────────────────────────

mcp = FastMCP(
    "PDPA-RAG",
    host=HOST,
    port=PORT,
    stateless_http=True,
    json_response=True,
)

# ─── Tool 1: search_pdpa ──────────────────────────────────


@mcp.tool()
def search_pdpa(
    query: str,
    top_k: int = 5,
) -> str:
    """ค้นหาความรู้จาก PDPA (พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562)
    คืน context จาก knowledge graph + เนื้อหามาตราต้นฉบับ

    ใช้เป็น tool แรกเสมอ สำหรับทุกคำถามเกี่ยวกับ PDPA
    หลังจากได้ผลลัพธ์แล้ว ให้ดูมาตราที่พบ แล้วเรียก get_related_sections
    เพื่อค้นหามาตราเชื่อมโยงเพิ่มเติม โดยเฉพาะ:
    - ข้อยกเว้นและเงื่อนไขที่อาจอยู่ในมาตราอื่น
    - บทลงโทษที่เชื่อมกับมาตราหลัก
    - สิทธิ หน้าที่ และหลักการที่เกี่ยวข้อง

    ตัวอย่าง query ที่ดี: 'เก็บข้อมูลชีวภาพ ลายนิ้วมือ ข้อมูลอ่อนไหว' หรือ
    'โทษจำคุก บทลงโทษทางอาญา' หรือ 'DPO รายงานเหตุ data breach แจ้งเตือน'
    """
    try:
        result = rag.search(query, top_k=top_k)

        if "error" in result:
            return json.dumps(
                {"tool": "search_pdpa", "error": result["error"], "context": ""},
                ensure_ascii=False,
            )

        context = result.get("context", "")
        keywords = result.get("keywords", {})
        profile_count = len(result.get("profile_results", []))
        chunk_count = len(result.get("chunk_results", []))

        response = {
            "tool": "search_pdpa",
            "query": query,
            "keywords": keywords,
            "profile_results_count": profile_count,
            "chunk_results_count": chunk_count,
            "context": context,
        }

        if not context.strip():
            response["message"] = "ไม่พบข้อมูลที่เกี่ยวข้องในระบบ"

        return json.dumps(response, ensure_ascii=False)

    except Exception as e:
        return json.dumps(
            {"tool": "search_pdpa", "error": str(e), "context": ""},
            ensure_ascii=False,
        )


# ─── Tool 2: get_related_sections ─────────────────────────


@mcp.tool()
def get_related_sections(section_id: str) -> str:
    """หามาตราและ entities ที่เชื่อมโยงกับมาตราที่ระบุ ผ่าน Knowledge Graph

    ★ สำคัญ: ควรเรียก tool นี้เสมอหลังจาก search_pdpa เพื่อให้ได้คำตอบที่ครบถ้วน
    โดยเรียกกับมาตราหลักที่พบจาก search เช่น ถ้า search พบ มาตรา 26
    ให้เรียก get_related_sections('sec_26') เพื่อจะเห็น:
    - มาตราที่อ้างอิงถึง (เช่น ม.27, ม.28 ที่เกี่ยวกับการใช้/เปิดเผย)
    - บทลงโทษที่เชื่อมโยง (เช่น ม.79 โทษอาญา, ม.84 โทษปกครอง)
    - ข้อยกเว้นที่ซ่อนอยู่ในมาตราอื่น
    - สิทธิ หน้าที่ หลักการที่เชื่อมโยง

    section_id ใช้รูปแบบ: sec_26, sec_37, sec_79 (sec_ + เลขมาตรา)

    ตัวอย่างการใช้งาน:
    - คำถามเรื่องข้อมูลอ่อนไหว → get_related_sections('sec_26')
    - คำถามเรื่อง breach notification → get_related_sections('sec_37')
    - คำถามเรื่องโทษอาญา → get_related_sections('sec_79')
    """
    try:
        result = rag.find_related_sections(section_id)
        result["tool"] = "get_related_sections"
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"tool": "get_related_sections", "error": str(e)}, ensure_ascii=False)


# ─── Tool 3: get_penalty ──────────────────────────────────


@mcp.tool()
def get_penalty(section_id: str) -> str:
    """หาบทลงโทษที่เกี่ยวข้องกับมาตราที่ระบุ โดยค้นผ่าน knowledge graph
    คืนทั้งโทษอาญา (จำคุก/ปรับ) และโทษปรับทางปกครอง พร้อม path การเชื่อมโยง

    ★ ควรเรียก tool นี้ทุกครั้งที่คำถามเกี่ยวกับ:
    - 'มีโทษอะไร' 'โทษเท่าไหร่' 'ผิดอะไร'
    - การฝ่าฝืน/ไม่ปฏิบัติตามมาตราใดมาตราหนึ่ง
    - บทลงโทษสูงสุด/ต่ำสุดของ PDPA

    section_id ใช้มาตราที่ถูกฝ่าฝืน ไม่ใช่มาตราโทษ เช่น:
    - ถ้าถามว่า 'ฝ่าฝืน ม.26 โทษอะไร' → get_penalty('sec_26') ✓
    - ไม่ใช่ get_penalty('sec_84') ✗ (ม.84 เป็นมาตราโทษเอง จะ return 0)

    ตัวอย่างการใช้:
    - เก็บข้อมูลอ่อนไหวไม่ขอ consent → get_penalty('sec_26')
    - ไม่แจ้ง breach → get_penalty('sec_37')
    - ใช้ข้อมูลผิดวัตถุประสงค์ → get_penalty('sec_27')
    """
    try:
        result = rag.get_penalty_for_section(section_id)
        result["tool"] = "get_penalty"
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"tool": "get_penalty", "error": str(e)}, ensure_ascii=False)


# ─── Tool 4: get_pdpa_summary ─────────────────────────────


@mcp.tool()
def get_pdpa_summary() -> str:
    """แสดงสรุปภาพรวมของ PDPA ทั้งฉบับจากระบบ RAG
    คืน: จำนวน nodes/edges ใน knowledge graph, ประเภท entities, สถิติมาตรา

    ใช้เมื่อคำถามถามภาพรวมของ PDPA เช่น:
    - "PDPA มีกี่มาตรา" "มีกี่หมวด"
    - "สรุปโครงสร้างของ PDPA"
    - "PDPA ครอบคลุมเรื่องอะไรบ้าง"
    - "ระบบ RAG มีข้อมูลอะไรบ้าง"

    ไม่ต้องใช้สำหรับคำถามเฉพาะเจาะจง (ใช้ search_pdpa แทน)"""
    try:
        summary = rag.get_summary()

        if "error" in summary:
            return json.dumps(
                {"tool": "get_pdpa_summary", "error": summary["error"]},
                ensure_ascii=False,
            )

        summary["tool"] = "get_pdpa_summary"
        return json.dumps(summary, ensure_ascii=False)

    except Exception as e:
        return json.dumps(
            {"tool": "get_pdpa_summary", "error": str(e)},
            ensure_ascii=False,
        )


# ─── Tool 5: get_section_text ──────────────────


@mcp.tool()
def get_section_text(section_id: str) -> str:
    """ดึง "ตัวบทต้นฉบับเต็ม" ของมาตรา PDPA ที่ระบุ แบบเจาะจง section_id

    ★ ใช้ tool นี้ทุกครั้งก่อนจะ "อ้างอิงเลขมาตราหรือเนื้อหากฎหมายในคำตอบ"
    เมื่อ search_pdpa หรือ get_related_sections ชี้ไปยังมาตราใด
    (เช่น เห็น edge sec_27 → sec_39) ให้เรียก get_section_text('sec_39')
    เพื่ออ่านตัวบทจริงก่อนสรุป — ห้ามเติมตัวบทจากความจำ

    ต่างจาก search_pdpa: tool นี้คืนตัวบท "เป๊ะตาม section_id" เสมอ
    (deterministic) ไม่ใช่ผลค้นเชิงความหมายที่อาจได้มาตราอื่น

    section_id ใช้รูปแบบ sec_<เลขมาตรา> เช่น get_section_text('sec_27')
    """
    try:
        result = rag.get_section_text(section_id)
        result["tool"] = "get_section_text"
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps(
            {"tool": "get_section_text", "error": str(e)},
            ensure_ascii=False,
        )


# ─── Run ───────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n🚀 MCP Server PDPA RAG starting on {HOST}:{PORT}/mcp")
    print(f"   Transport: streamable-http")
    print(f"   Tools: 5 (search_pdpa, get_related_sections, get_section_text, get_penalty, get_pdpa_summary)")
    mcp.run(transport="streamable-http")
