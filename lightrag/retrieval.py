"""
Retrieval — PDPA Legal Knowledge Retrieval (PDPA RAG version)
=============================================================
keyword extraction → vector search → multi-hop BFS → hybrid context
ปรับ prompts จาก XXX ระบบไฟฟ้า → PDPA กฎหมายคุ้มครองข้อมูลส่วนบุคคล
"""

import networkx as nx
from .llm_provider import query_llm_json, query_llm
from .vector_index import VectorIndex

KEYWORD_PROMPT = """จากคำถามนี้: "{query}"

สร้าง keywords สำหรับค้นหาในกฎหมาย PDPA (พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล) ตอบเป็น JSON:
{{
  "low_level_keywords": ["keywords เฉพาะเจาะจง เช่น เลขมาตรา ชื่อหน่วยงาน สิทธิเฉพาะ บทลงโทษ"],
  "high_level_keywords": ["keywords กว้าง เช่น หลักการ หมวดหมู่ แนวคิด ขั้นตอน"]
}}

ตัวอย่าง:
คำถาม: "เก็บข้อมูลสุขภาพพนักงานได้ไหม"
→ low: ["ข้อมูลสุขภาพ", "มาตรา ๒๖", "ข้อมูลอ่อนไหว", "ความยินยอมโดยชัดแจ้ง"]
→ high: ["ฐานทางกฎหมาย", "การเก็บรวบรวม", "sensitive data", "ข้อยกเว้น"]

คำถาม: "ส่งข้อมูลลูกค้าไปต่างประเทศต้องทำอย่างไร"
→ low: ["การส่งข้อมูลข้ามประเทศ", "มาตรา ๒๘", "มาตรา ๒๙", "มาตรฐานการคุ้มครอง"]
→ high: ["cross-border transfer", "ความเพียงพอของการคุ้มครอง", "ข้อยกเว้น"]

ตอบ JSON เท่านั้น"""


def extract_query_keywords(query: str) -> dict:
    prompt = KEYWORD_PROMPT.format(query=query)
    result = query_llm_json(
        prompt,
        system_prompt="สร้าง keywords สำหรับค้นหากฎหมาย PDPA ตอบ JSON เท่านั้น"
    )
    low = result.get("low_level_keywords", [])
    high = result.get("high_level_keywords", [])
    if not low and not high:
        low = query.split()
    return {"low_level": low, "high_level": high}


def retrieve(query: str, index: VectorIndex,
             knowledge_graph: nx.DiGraph,
             top_k: int = 5, max_chunks: int = 5) -> dict:
    """
    PDPA retrieval pipeline:
    1. LLM extract keywords จากคำถาม
    2. Vector search profiles (graph nodes/edges)
    3. Vector search chunks (เนื้อหามาตรา)
    4. Multi-hop BFS expansion จาก graph
    5. Build hybrid context
    """
    keywords = extract_query_keywords(query)
    low_query = " ".join(keywords["low_level"])
    high_query = " ".join(keywords["high_level"])

    # Profile search (knowledge graph)
    profile_results = []
    if high_query.strip():
        profile_results = index.search_profiles(high_query, top_k=top_k)
    if low_query.strip():
        more = index.search_profiles(low_query, top_k=3)
        seen = {r["profile"]["id"] for r in profile_results}
        for r in more:
            if r["profile"]["id"] not in seen:
                profile_results.append(r)

    # Multi-hop BFS expansion
    expanded = _expand_from_graph(profile_results, knowledge_graph, max_expand=3, max_hops=2)

    # Chunk search (มาตรา text)
    chunk_query = f"{low_query} {high_query}".strip() or query
    chunk_results = index.search_chunks(chunk_query, top_k=max_chunks)

    context = _build_context(profile_results, expanded, chunk_results)

    return {
        "keywords": keywords,
        "profile_results": profile_results,
        "chunk_results": chunk_results,
        "context": context,
    }


def _expand_from_graph(results: list[dict], G: nx.DiGraph,
                       max_expand: int = 3, max_hops: int = 2) -> list[str]:
    """Multi-hop BFS expansion จาก knowledge graph"""
    expanded = []
    visited = set()

    seed_ids = []
    for r in results[:max_expand]:
        node_id = r["profile"]["id"]
        if "--" in node_id:
            continue
        if G.has_node(node_id):
            seed_ids.append(node_id)

    from collections import deque
    queue = deque()
    for sid in seed_ids:
        visited.add(sid)
        queue.append((sid, 0))

    while queue:
        current, hop = queue.popleft()
        if hop >= max_hops:
            continue

        neighbors = list(G.successors(current)) + list(G.predecessors(current))
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            visited.add(neighbor)

            n_data = G.nodes.get(neighbor, {})
            edge_out = G.edges.get((current, neighbor), {})
            edge_in = G.edges.get((neighbor, current), {})
            edge = edge_out or edge_in

            desc = n_data.get("description", n_data.get("label", neighbor))
            rel = edge.get("relation", edge.get("relationship", edge.get("type", "related")))

            hop_label = f"hop{hop+1}"
            expanded.append(f"[{hop_label}][{rel}] {neighbor}: {desc}")

            queue.append((neighbor, hop + 1))

    return expanded


def _build_context(profile_results, expanded, chunk_results) -> str:
    """สร้าง hybrid context จาก graph profiles + chunks"""
    ctx = ""

    if profile_results:
        ctx += "## ความรู้จาก Knowledge Graph (PDPA)\n"
        for r in profile_results[:8]:
            ctx += f"- {r['profile']['value']}\n"
        if expanded:
            ctx += "\n### ข้อมูลเชื่อมโยง (multi-hop graph):\n"
            for e in expanded[:15]:
                ctx += f"- {e}\n"

    if chunk_results:
        ctx += "\n## เนื้อหาต้นฉบับ PDPA (Original Text)\n"
        for cr in chunk_results:
            section = cr.get("section_id", "")
            h2 = cr.get("header_2", "")
            text = cr.get("text", "")
            if text:
                header = f"{h2} — {section}" if h2 else section
                ctx += f"\n### {header}:\n{text}\n"

    return ctx


def generate_answer(query: str, context: str) -> str:
    """สร้างคำตอบจาก context"""
    prompt = f"""ใช้ข้อมูลต่อไปนี้ในการตอบคำถามเกี่ยวกับ PDPA:

{context}

คำถาม: {query}

กฎ:
- ตอบเป็นภาษาไทย กระชับ ได้ใจความ
- อ้างอิงมาตราที่เกี่ยวข้องให้ชัดเจน (เช่น "ตามมาตรา ๒๖")
- แยกให้ชัดระหว่าง "บทบัญญัติตามกฎหมาย" กับ "คำแนะนำเพิ่มเติม"
- ถ้ามีบทลงโทษที่เกี่ยวข้อง ให้ระบุด้วย
- ถ้าข้อมูลไม่เพียงพอให้บอกตรงๆ"""

    system = ("คุณเป็น AI ที่ปรึกษากฎหมาย PDPA (พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562) "
              "ตอบคำถามโดยอ้างอิงจากเนื้อหากฎหมายจริง")
    return query_llm(prompt, system_prompt=system)
