#!/usr/bin/env python3
"""Validate a PDPA knowledge-graph JSON from the data itself (never trust the file's metadata block).

Usage:
    python scripts/validate_graph.py                       # checks data/pdpa_knowledge_graph.json
    python scripts/validate_graph.py build/pdpa_knowledge_graph.json
    python scripts/validate_graph.py --penalty sec_26      # can sec_26 reach a Penalty node?

Exit code is non-zero when a check fails, so it can run in CI.
"""
import argparse, json, os, sys
from collections import deque
import networkx as nx

p = argparse.ArgumentParser()
p.add_argument("path", nargs="?", default=os.path.join(os.path.dirname(__file__), "..", "data", "pdpa_knowledge_graph.json"))
p.add_argument("--sections", type=int, default=96, help="expected number of Section nodes")
p.add_argument("--penalty", default="sec_26", help="section id that must reach a Penalty node within --hops")
p.add_argument("--hops", type=int, default=3)
a = p.parse_args()

g = json.load(open(a.path, encoding="utf-8"))
G = nx.DiGraph()
for n in g["nodes"]:
    G.add_node(n["id"], **n)
for e in g["edges"]:
    G.add_edge(e["source"], e["target"], relationship=e["relationship"])

meta = g.get("metadata", {})
sections = sum(1 for _, d in G.nodes(data=True) if d.get("label") == "Section")
hubs = sorted(((n, d) for n, d in G.degree if G.nodes[n].get("label") == "Section"), key=lambda x: -x[1])[:3]

def reaches_penalty(start, hops):
    seen, q = {start}, deque([(start, 0)])
    while q:
        cur, h = q.popleft()
        if G.nodes[cur].get("label") == "Penalty":
            return True
        if h >= hops:
            continue
        for nb in list(G.successors(cur)) + list(G.predecessors(cur)):
            if nb not in seen:
                seen.add(nb); q.append((nb, h + 1))
    return False

checks = {
    "nodes/edges (counted)": (f"{G.number_of_nodes()} / {G.number_of_edges()}"
                              + (f"   (metadata block says {meta.get('node_count')} / {meta.get('edge_count')})" if meta else ""), True),
    f"Section nodes == {a.sections}": (sections, sections == a.sections),
    "single weakly connected component": (nx.is_weakly_connected(G), nx.is_weakly_connected(G)),
    f"{a.penalty} reaches a Penalty node within {a.hops} hops": (reaches_penalty(a.penalty, a.hops), reaches_penalty(a.penalty, a.hops)),
    "top-3 Section hubs (node, degree)": (hubs, True),
}
ok = True
for name, (value, passed) in checks.items():
    print(f"[{'OK' if passed else 'FAIL'}] {name}: {value}")
    ok &= passed
sys.exit(0 if ok else 1)
