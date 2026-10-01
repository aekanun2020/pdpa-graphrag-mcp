"""
Knowledge Graph — สร้างจาก PDPA Markdown + LLM extraction (PDPA RAG version)
=============================================================================
- ใช้ MarkdownNodeParser-style chunking (split ตาม heading hierarchy)
- LLM extract legal entities/relations จากแต่ละมาตรา
- Load pre-built rule-based graph จาก pdpa_knowledge_graph.json
"""

import os
import re
import json
import networkx as nx
from .llm_provider import query_llm_json, query_llm

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
PDPA_MD = os.path.join(DATA_DIR, "pdpa_full_text_v2.md")
PDPA_GRAPH_JSON = os.path.join(DATA_DIR, "pdpa_knowledge_graph.json")


# ─── Markdown Chunking (MarkdownNodeParser-style) ─────────

def chunk_markdown(filepath: str) -> tuple[list[dict], dict]:
    """
    Parse Markdown ตาม heading hierarchy → 1 chunk = 1 มาตรา
    คืน (chunks, chunk_store)

    chunks: list ของ dict สำหรับ LLM extraction
    chunk_store: dict สำหรับ vector indexing (chunk_id → {text, metadata})
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = []
    chunk_store = {}

    # Track current heading context
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""
    current_h4 = ""
    current_body = []

    def flush_section():
        """Save accumulated body text as a chunk"""
        if not current_h4 or not current_body:
            return
        body_text = "\n".join(current_body).strip()
        if len(body_text) < 20:
            return

        # Extract section number from heading (e.g. "มาตรา ๑" → 1)
        section_id = current_h4.strip()

        chunk = {
            "section_id": section_id,
            "header_1": current_h1,
            "header_2": current_h2,
            "header_3": current_h3,
            "header_4": current_h4,
            "text": body_text,
        }
        chunks.append(chunk)

        # chunk_store key
        safe_id = re.sub(r'\s+', '-', section_id)
        chunk_store[safe_id] = {
            "text": f"[{current_h2}] [{current_h3}] {section_id}\n{body_text}" if current_h3 else f"[{current_h2}] {section_id}\n{body_text}",
            "section_id": section_id,
            "header_2": current_h2,
            "header_3": current_h3,
        }

    for line in content.split("\n"):
        stripped = line.strip()

        # Detect heading levels
        if stripped.startswith("#### "):
            flush_section()
            current_h4 = stripped[5:].strip()
            current_body = []
        elif stripped.startswith("### "):
            flush_section()
            current_h3 = stripped[4:].strip()
            current_h4 = ""
            current_body = []
        elif stripped.startswith("## "):
            flush_section()
            current_h2 = stripped[3:].strip()
            current_h3 = ""
            current_h4 = ""
            current_body = []
        elif stripped.startswith("# ") and not stripped.startswith("## "):
            current_h1 = stripped[2:].strip()
        else:
            if current_h4:
                current_body.append(line)

    # Flush last section
    flush_section()

    return chunks, chunk_store


# ─── LLM Entity Extraction ──────────────────────────────

EXTRACT_PROMPT = """จากเนื้อหากฎหมาย PDPA (พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล) ต่อไปนี้:

{section_header}
{chunk}

ให้ extract entities และ relationships ที่เกี่ยวข้อง ตอบเป็น JSON เท่านั้น:
{{
  "entities": [
    {{"name": "ชื่อ entity", "type": "ประเภท", "description": "คำอธิบายสั้นๆ"}}
  ],
  "relations": [
    {{"source": "entity ต้นทาง", "target": "entity ปลายทาง", "relation": "ความสัมพันธ์", "description": "คำอธิบาย"}}
  ]
}}

กฎ:
- ประเภท entity ที่ควรดึง: LegalSection, Right, Obligation, Penalty, Definition, LawfulBasis, Organization, Role, DataCategory, Condition, Exemption, Procedure
- relation ควรเป็นวลีสั้นๆ เช่น "grants_right", "imposes_obligation", "prescribes_penalty", "defines", "requires_consent", "exempts", "references"
- ดึงเฉพาะข้อมูลที่มีในข้อความจริง ห้ามสร้างขึ้นมาเอง
- ใส่ชื่อมาตราที่เกี่ยวข้องเป็น entity ด้วย (เช่น "มาตรา ๒๖")"""


def extract_from_chunk(chunk: dict) -> dict:
    """LLM extract entities/relations จาก 1 มาตรา"""
    section_header = f"[{chunk['header_2']}] {chunk['section_id']}" if chunk.get('header_2') else chunk['section_id']
    prompt = EXTRACT_PROMPT.format(
        section_header=section_header,
        chunk=chunk["text"][:3000],  # cap at ~3000 chars
    )
    result = query_llm_json(
        prompt,
        system_prompt="คุณเป็นผู้เชี่ยวชาญกฎหมายคุ้มครองข้อมูลส่วนบุคคล extract entities และ relationships จาก PDPA ตอบเป็น JSON เท่านั้น"
    )
    section_id = chunk.get("section_id", "")
    if "entities" in result:
        for e in result["entities"]:
            e["source_section"] = section_id
    if "relations" in result:
        for r in result["relations"]:
            r["source_section"] = section_id
    return result


def build_knowledge_graph_from_extractions(extractions: list[dict]) -> nx.DiGraph:
    """สร้าง Knowledge Graph จาก LLM extractions"""
    G = nx.DiGraph()
    for ext in extractions:
        if "error" in ext:
            continue
        for e in ext.get("entities", []):
            name = e.get("name", "").strip()
            if not name:
                continue
            if G.has_node(name):
                existing = G.nodes[name].get("description", "")
                new_desc = e.get("description", "")
                if new_desc and new_desc not in existing:
                    G.nodes[name]["description"] = f"{existing}; {new_desc}"
                sources = G.nodes[name].get("source_sections", set())
                sources.add(e.get("source_section", ""))
                G.nodes[name]["source_sections"] = sources
            else:
                G.add_node(name, **{
                    "type": e.get("type", "Unknown"),
                    "description": e.get("description", ""),
                    "node_category": "knowledge", "graph_source": "knowledge",
                    "source_sections": {e.get("source_section", "")},
                })
        for r in ext.get("relations", []):
            src = r.get("source", "").strip()
            tgt = r.get("target", "").strip()
            rel = r.get("relation", "").strip()
            if not (src and tgt and rel):
                continue
            for n in [src, tgt]:
                if not G.has_node(n):
                    G.add_node(n, type="Unknown", description="",
                               node_category="knowledge", graph_source="knowledge",
                               source_sections={r.get("source_section", "")})
            if not G.has_edge(src, tgt):
                G.add_edge(src, tgt, relation=rel,
                           description=r.get("description", ""),
                           source_section=r.get("source_section", ""))
    return G


# ─── Pre-built Rule-based Graph ──────────────────────────

def load_prebuilt_graph(filepath: str) -> nx.DiGraph:
    """Load pre-built PDPA knowledge graph จาก JSON (163 nodes, 343 edges)"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.DiGraph()
    for node in data.get("nodes", []):
        node_id = node.get("id", "")
        attrs = {k: v for k, v in node.items() if k != "id"}
        attrs["node_category"] = "prebuilt"
        attrs["graph_source"] = "prebuilt"
        G.add_node(node_id, **attrs)

    for edge in data.get("edges", []):
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
        G.add_edge(src, tgt, **attrs)

    return G


def profile_knowledge_graph(G: nx.DiGraph, use_llm: bool = False) -> list[dict]:
    """สร้าง search profiles จาก graph nodes"""
    profiles = []
    for node, data in G.nodes(data=True):
        rels = []
        for u, v, edata in G.edges(data=True):
            if u == node:
                rels.append(f"{edata.get('relation', edata.get('relationship', edata.get('type', '')))}: {v}")
            elif v == node:
                rels.append(f"{u} {edata.get('relation', edata.get('relationship', edata.get('type', '')))}")

        if use_llm:
            prompt = PROFILE_PROMPT.format(
                name=node, etype=data.get("type", ""),
                description=data.get("description", data.get("label", "")),
                relations="; ".join(rels[:8]),
            )
            kw_result = query_llm_json(
                prompt,
                system_prompt="คุณเป็นผู้เชี่ยวชาญกฎหมาย PDPA สร้าง keywords สำหรับ search ตอบ JSON เท่านั้น"
            )
            low_keys = kw_result.get("low_level_keywords", [node])
            high_keys = kw_result.get("high_level_keywords", [])
            all_keys = low_keys + high_keys
        else:
            all_keys = [node]
            node_type = data.get("type", "")
            label = data.get("label", "")
            desc = data.get("description", "")
            if node_type:
                all_keys.append(node_type)
            if label and label != node:
                all_keys.append(label)
            if desc:
                all_keys.append(desc[:200])
            all_keys += [r.split(":")[0].strip() for r in rels[:5]]

        all_keys = [k for k in all_keys if k and str(k).strip()]
        desc_text = data.get("description", data.get("label", ""))
        value = (f"{node} ({data.get('type', '')}): {desc_text}\n"
                 f"ความสัมพันธ์: {'; '.join(rels[:8])}")
        profiles.append({
            "id": node, "keys": all_keys, "value": value,
            "node_category": data.get("node_category", "knowledge"),
        })

    # Edge profiles
    for u, v, data in G.edges(data=True):
        rel = data.get("relation", data.get("relationship", data.get("type", "")))
        desc = data.get("description", "")
        if rel:
            profiles.append({
                "id": f"{u}--{rel}--{v}",
                "keys": [rel, u, v] + ([desc[:100]] if desc else []),
                "value": f"{u} → [{rel}] → {v}: {desc}" if desc else f"{u} → [{rel}] → {v}",
                "node_category": "knowledge_relation",
            })
    return profiles


PROFILE_PROMPT = """จาก entity นี้ในบริบทกฎหมาย PDPA:

ชื่อ: {name}
ประเภท: {etype}
คำอธิบาย: {description}
ความสัมพันธ์: {relations}

สร้าง keywords สำหรับค้นหา ตอบเป็น JSON:
{{
  "low_level_keywords": ["keyword เฉพาะเจาะจง เช่น ชื่อมาตรา หน่วยงาน สิทธิเฉพาะ"],
  "high_level_keywords": ["keyword กว้างๆ เช่น หลักการ แนวคิด หมวดหมู่"]
}}"""


# ─── Main Pipeline ───────────────────────────────────────

def ingest_pdpa(use_llm_extract: bool = True, use_llm_profile: bool = False) -> tuple[nx.DiGraph, list[dict], dict]:
    """
    Pipeline หลัก:
    1. Load pre-built rule-based graph (163 nodes, 343 edges)
    2. Parse PDPA markdown → chunks (1 chunk = 1 มาตรา)
    3. (Optional) LLM extract entities → merge เข้า graph
    4. Profile graph nodes → search profiles
    5. คืน (graph, profiles, chunk_store)
    """
    # [1] Load pre-built graph
    print(f"  Loading pre-built PDPA graph from {PDPA_GRAPH_JSON}...")
    prebuilt = load_prebuilt_graph(PDPA_GRAPH_JSON)
    print(f"  ✅ Pre-built: {prebuilt.number_of_nodes()} nodes, {prebuilt.number_of_edges()} edges")

    # [2] Parse markdown → chunks
    print(f"  Parsing PDPA markdown from {PDPA_MD}...")
    chunks, chunk_store = chunk_markdown(PDPA_MD)
    print(f"  ✅ Parsed {len(chunks)} sections (มาตรา), {len(chunk_store)} chunks")

    # [3] LLM extract (optional — costs API calls)
    if use_llm_extract:
        print(f"  LLM extracting entities from {len(chunks)} sections...")
        extractions = []
        for i, chunk in enumerate(chunks):
            print(f"    [{i+1}/{len(chunks)}] {chunk['section_id']}...", end=" ", flush=True)
            ext = extract_from_chunk(chunk)
            n_ent = len(ext.get("entities", []))
            n_rel = len(ext.get("relations", []))
            print(f"→ {n_ent} entities, {n_rel} relations")
            extractions.append(ext)

        llm_graph = build_knowledge_graph_from_extractions(extractions)
        print(f"  ✅ LLM graph: {llm_graph.number_of_nodes()} nodes, {llm_graph.number_of_edges()} edges")

        # Merge LLM graph into pre-built graph
        merged = nx.compose(prebuilt, llm_graph)
        print(f"  ✅ Merged: {merged.number_of_nodes()} nodes, {merged.number_of_edges()} edges")
    else:
        merged = prebuilt
        print(f"  ⏭️  Skipping LLM extract (use_llm_extract=False)")

    # [4] Profile
    print(f"  Profiling {merged.number_of_nodes()} nodes...")
    profiles = profile_knowledge_graph(merged, use_llm=use_llm_profile)
    print(f"  ✅ {len(profiles)} profiles")

    return merged, profiles, chunk_store
