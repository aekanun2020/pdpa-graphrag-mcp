"""
PDPA LightRAG — Main Library สำหรับ MCP Server
================================================
Pipeline: Load pre-built graph → Parse MD → Embed → Index → Retrieve
ไม่มี Fact Graph (ต่างจาก XXX) — ใช้ single knowledge graph
"""

import sys
from .knowledge_graph import ingest_pdpa
from .vector_index import VectorIndex
from .retrieval import retrieve, generate_answer


class PDPALightRAG:
    def __init__(self):
        self.knowledge_graph = None
        self.index = None
        self.chunk_store = {}
        self._initialized = False

    def build(self, use_llm_extract: bool = False, use_llm_profile: bool = False):
        """
        สร้างระบบ PDPA RAG

        Args:
            use_llm_extract: True = LLM extract entities จากแต่ละมาตรา (ช้า+ใช้ API)
                             False = ใช้ pre-built graph อย่างเดียว (เร็ว)
            use_llm_profile: True = LLM สร้าง keywords profile (ช้า+ใช้ API)
                             False = rule-based profile (เร็ว)
        """
        print("=" * 60)
        print("PDPA LightRAG: กำลังสร้างระบบ...")
        print("=" * 60)

        print("\n[1/3] สร้าง Knowledge Graph จาก PDPA...")
        self.knowledge_graph, kg_profiles, self.chunk_store = ingest_pdpa(
            use_llm_extract=use_llm_extract,
            use_llm_profile=use_llm_profile,
        )
        print(f"  ✅ {self.knowledge_graph.number_of_nodes()} nodes, "
              f"{self.knowledge_graph.number_of_edges()} edges")

        print("\n[2/3] สร้าง Vector Index (BGE-M3 + Qdrant)...")
        self.index = VectorIndex()
        self.index.build_profiles(kg_profiles)
        print(f"  ✅ {len(kg_profiles)} profiles indexed")

        print("\n[3/3] Index มาตรา chunks...")
        self.index.build_chunks(self.chunk_store)
        print(f"  ✅ {len(self.chunk_store)} chunks indexed")

        self._initialized = True
        self._profile_count = len(kg_profiles)
        self._chunk_count = len(self.chunk_store)
        print(f"\n{'='*60}")
        print("PDPA LightRAG พร้อมใช้งาน!")
        print(f"  Knowledge Graph: {self.knowledge_graph.number_of_nodes()} nodes, "
              f"{self.knowledge_graph.number_of_edges()} edges")
        print(f"  Profiles: {self._profile_count}")
        print(f"  Chunks: {self._chunk_count}")
        print(f"{'='*60}")
        sys.stdout.flush()

    def search(self, query: str, top_k: int = 5) -> dict:
        """ค้นหาแล้วคืน raw result สำหรับ MCP tool"""
        if not self._initialized:
            return {"error": "PDPA LightRAG ยังไม่ได้ build"}
        return retrieve(query, self.index, self.knowledge_graph, top_k=top_k)

    def query(self, question: str, top_k: int = 5) -> str:
        """ค้นหา + สร้างคำตอบ (standalone test)"""
        if not self._initialized:
            return "[ERROR] กรุณาเรียก .build() ก่อน"
        result = retrieve(question, self.index, self.knowledge_graph, top_k=top_k)
        return generate_answer(question, result["context"])

    # ─── Graph Traversal Methods ─────────────────────────────

    def find_related_sections(self, section_id: str) -> dict:
        """หามาตราที่เกี่ยวข้องกับมาตราที่ระบุ ผ่าน graph traversal"""
        if not self._initialized:
            return {"error": "PDPA LightRAG ยังไม่ได้ build"}

        G = self.knowledge_graph
        if not G.has_node(section_id):
            # Try fuzzy match
            candidates = [n for n in G.nodes() if section_id.lower() in n.lower()]
            if not candidates:
                return {"error": f"ไม่พบ '{section_id}' ใน Knowledge Graph",
                        "hint": "ลองใช้ search_pdpa แทน"}
            section_id = candidates[0]

        # Outgoing edges
        outgoing = []
        for u, v, edata in G.out_edges(section_id, data=True):
            v_data = G.nodes.get(v, {})
            outgoing.append({
                "target": v,
                "target_type": v_data.get("type", ""),
                "relation": edata.get("relation", edata.get("relationship", edata.get("type", ""))),
                "description": edata.get("description", v_data.get("description", "")),
            })

        # Incoming edges
        incoming = []
        for u, v, edata in G.in_edges(section_id, data=True):
            u_data = G.nodes.get(u, {})
            incoming.append({
                "source": u,
                "source_type": u_data.get("type", ""),
                "relation": edata.get("relation", edata.get("relationship", edata.get("type", ""))),
                "description": edata.get("description", u_data.get("description", "")),
            })

        node_data = G.nodes.get(section_id, {})
        return {
            "section_id": section_id,
            "section_type": node_data.get("type", ""),
            "description": node_data.get("description", node_data.get("label", "")),
            "outgoing_relations": outgoing,
            "incoming_relations": incoming,
            "total_connections": len(outgoing) + len(incoming),
        }

    def get_section_text(self, section_id: str) -> dict:
        """คืนตัวบทต้นฉบับเต็มของมาตราที่ระบุ (deterministic by-id lookup)

        ต่างจาก search() ตรงที่ "ไม่ทำ semantic match" — ค้น sec_27 ต้องได้
        ม.27 เสมอ ไม่มีวันได้มาตราอื่น เพื่อใช้เป็นแหล่ง grounding ที่แน่นอน
        """
        if not self._initialized:
            return {"error": "PDPA LightRAG ยังไม่ได้ build"}

        G = self.knowledge_graph

        # normalize: "27" -> "sec_27"
        sid = str(section_id).strip()
        if sid.isdigit():
            sid = f"sec_{sid}"

        # exact match ก่อนเสมอ (กัน substring ชนกัน เช่น "2" ไปชน sec_20/sec_27)
        if not G.has_node(sid):
            return {
                "section_id": section_id,
                "found": False,
                "error": f"ไม่พบมาตรา '{section_id}' ใน Knowledge Graph",
                "hint": "ใช้ search_pdpa ค้นหาก่อน แล้วเรียกซ้ำด้วย section_id ที่ได้",
            }

        node = G.nodes[sid]
        if node.get("label") != "Section":
            return {
                "section_id": section_id,
                "found": False,
                "error": f"'{sid}' ไม่ใช่ node ประเภท Section (เป็น {node.get('label')})",
            }

        text = (node.get("text") or "").strip()
        return {
            "section_id": sid,
            "number": node.get("number"),
            "text": text,
            "found": bool(text),
        }

    def get_penalty_for_section(self, section_id: str) -> dict:
        """หาบทลงโทษที่เกี่ยวข้องกับมาตราที่ระบุ"""
        if not self._initialized:
            return {"error": "PDPA LightRAG ยังไม่ได้ build"}

        G = self.knowledge_graph
        penalties = []

        # BFS: section → ... → Penalty nodes (max 3 hops)
        from collections import deque
        visited = set()
        queue = deque()

        # Find the section node (fuzzy match)
        target = None
        for n in G.nodes():
            if section_id.lower() in n.lower():
                target = n
                break
        if not target:
            return {"error": f"ไม่พบมาตรา '{section_id}'", "penalties": []}

        visited.add(target)
        queue.append((target, 0, [target]))

        while queue:
            current, hop, path = queue.popleft()
            if hop >= 3:
                continue

            node_data = G.nodes.get(current, {})
            if node_data.get("label") == "Penalty" and current != target:
                penalties.append({
                    "penalty_node": current,
                    "description": node_data.get("description", node_data.get("label", "")),
                    "path": " → ".join(path),
                    "hops": hop,
                })

            neighbors = list(G.successors(current)) + list(G.predecessors(current))
            for nb in neighbors:
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, hop + 1, path + [nb]))

        return {
            "section_id": section_id,
            "matched_node": target,
            "penalties_found": len(penalties),
            "penalties": penalties,
        }

    def get_summary(self) -> dict:
        """คืนสรุประบบ"""
        if not self._initialized:
            return {"error": "PDPA LightRAG ยังไม่ได้ build"}

        # Node type distribution
        type_counts = {}
        for _, d in self.knowledge_graph.nodes(data=True):
            t = d.get("label", d.get("type", "Unknown"))
            type_counts[t] = type_counts.get(t, 0) + 1

        # Edge type distribution
        edge_types = {}
        for _, _, d in self.knowledge_graph.edges(data=True):
            t = d.get("relation", d.get("relationship", d.get("type", "Unknown")))
            edge_types[t] = edge_types.get(t, 0) + 1

        return {
            "knowledge_graph": {
                "nodes": self.knowledge_graph.number_of_nodes(),
                "edges": self.knowledge_graph.number_of_edges(),
                "node_types": type_counts,
                "edge_types": edge_types,
            },
            "vector_index": {
                "profiles": self._profile_count,
                "chunks": self._chunk_count,
                "embedding_model": "BGE-M3",
                "vector_store": "Qdrant",
                "collections": ["pdpa_profiles", "pdpa_chunks"],
            },
        }
