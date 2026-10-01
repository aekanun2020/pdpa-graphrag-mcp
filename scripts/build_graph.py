#!/usr/bin/env python3
"""
Build PDPA Knowledge Graph from full-text source.

Reads pdpa_full_text.txt (raw OCR text from rongkat.go.th PDF),
extracts 96 sections with cross-references, and outputs:
  - pdpa_knowledge_graph.json  (main graph — used by LightRAG)
  - pdpa_networkx.json         (NetworkX node-link format)
  - pdpa_neo4j_import.cypher   (Neo4j import script)
  - nodes.csv / edges.csv      (tabular format)

Usage:
    python scripts/build_graph.py                          # defaults
    python scripts/build_graph.py --input /path/to/txt     # custom input
    python scripts/build_graph.py --output ./data          # custom output dir
"""

import re
import json
import csv
import os
import argparse

# ──────────────────────────────────────────────────────────────
# CLI Arguments
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Build PDPA Knowledge Graph")
parser.add_argument(
    "--input", "-i",
    default=os.path.join(os.path.dirname(__file__), "..", "data", "pdpa_full_text_v2.md"),
    help="Path to the statute Markdown (default: data/pdpa_full_text_v2.md)",
)
parser.add_argument(
    "--output", "-o",
    default=os.path.join(os.path.dirname(__file__), "..", "build"),
    help="Output directory (default: build/ — never data/, so the shipped graph is not overwritten)",
)
args = parser.parse_args()

INPUT_FILE = os.path.abspath(args.input)
OUTPUT_DIR = os.path.abspath(args.output)

if not os.path.isfile(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 1. READ & CLEAN FULL TEXT
# ──────────────────────────────────────────────────────────────
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw = f.read()

# Fix OCR artifacts (จ า -> จำ etc.)
replacements = {
    "จ ากัด": "จำกัด", "จ าเป็น": "จำเป็น", "ท าการ": "ทำการ",
    "ท าให้": "ทำให้", "กระท า": "กระทำ", "ด าเนิน": "ดำเนิน",
    "ก าหนด": "กำหนด", "อ านาจ": "อำนาจ", "ส านัก": "สำนัก",
    "ค าสั่ง": "คำสั่ง", "ต าแหน่ง": "ตำแหน่ง", "ค าแนะ": "คำแนะ",
    "ท าหน้าที่": "ทำหน้าที่", "ท านอง": "ทำนอง", "จ านวน": "จำนวน",
    "ท าลาย": "ทำลาย", "ค าพิพากษา": "คำพิพากษา", "ค าร้อง": "คำร้อง",
    "ค าขอ": "คำขอ", "ค าวินิจฉัย": "คำวินิจฉัย", "จ าคุก": "จำคุก",
    "จ าเลย": "จำเลย", "ก าไร": "กำไร", "ส าเนา": "สำเนา",
    "ค านวณ": "คำนวณ", "ส าคัญ": "สำคัญ", "ด ารง": "ดำรง",
    "ส าหรับ": "สำหรับ", "อ านวย": "อำนวย", "น า": "นำ",
    "ซ้ า": "ซ้ำ", "ค านึง": "คำนึง", "ส ารวจ": "สำรวจ",
    "ส าเร็จ": "สำเร็จ", "ค าปรึกษา": "คำปรึกษา",
}
for old, new in replacements.items():
    raw = raw.replace(old, new)

# Remove page headers/footers
raw = re.sub(r'้หนา\s+\d+.*?๒๕๖๒', '', raw)
raw = re.sub(r'--- PAGE \d+ ---', '', raw)
# Collapse whitespace
raw = re.sub(r'\n\s*\n', '\n', raw)


def thai_to_arabic(thai_num):
    thai_digits = '๐๑๒๓๔๕๖๗๘๙'
    return ''.join(str(thai_digits.index(c)) if c in thai_digits else c for c in thai_num)


# ──────────────────────────────────────────────────────────────
# 2. EXTRACT SECTIONS
# ──────────────────────────────────────────────────────────────
# [LAB3 FIX] ไฟล์ v2 เป็น Markdown: มาตราอยู่ใน heading '#### มาตรา ๑'
# และตัวเลขไทยอาจตามด้วยช่องว่างหลายช่อง จึงจับจาก heading โดยตรง (แม่นยำกว่า raw regex เดิม)
header_pattern = re.compile(r'^#{1,6}\s*มาตรา\s+([\d๐-๙]+)', re.MULTILINE)

# Find all header positions
headers = []
for m in header_pattern.finditer(raw):
    sec_num = int(thai_to_arabic(m.group(1)))
    if 1 <= sec_num <= 96:
        headers.append((sec_num, m.start()))

# Deduplicate: keep first occurrence per section number
seen_nums = {}
unique_headers = []
for num, pos in headers:
    if num not in seen_nums:
        seen_nums[num] = pos
        unique_headers.append((num, pos))

unique_headers.sort(key=lambda x: x[1])

# Extract text between consecutive headers
sections = {}
for i, (num, pos) in enumerate(unique_headers):
    next_pos = unique_headers[i + 1][1] if i + 1 < len(unique_headers) else len(raw)
    text = raw[pos:next_pos].strip()
    text = re.sub(r'\s+', ' ', text)
    if num not in sections or len(text) > len(sections[num]):
        sections[num] = text

print(f"Extracted {len(sections)} unique sections")

# ──────────────────────────────────────────────────────────────
# 3. FIND CROSS-REFERENCES
# ──────────────────────────────────────────────────────────────
cross_refs = {}
for sec_num, text in sections.items():
    # [LAB3 FIX] รองรับช่องว่างหลายช่องระหว่าง 'มาตรา' กับตัวเลข
    refs = re.findall(r'มาตรา\s+([\d๐-๙]+)', text)
    other_refs = set()
    for r in refs:
        n = int(thai_to_arabic(r))
        if n != sec_num and 1 <= n <= 96 and n in sections:
            other_refs.add(n)
    if other_refs:
        cross_refs[sec_num] = sorted(other_refs)

print(f"Found {sum(len(v) for v in cross_refs.values())} cross-reference edges across {len(cross_refs)} sections")

for sec_num in sorted(list(cross_refs.keys()))[:15]:
    print(f"  sec_{sec_num} references -> {cross_refs[sec_num]}")

# ──────────────────────────────────────────────────────────────
# 4. BUILD GRAPH DATA
# ──────────────────────────────────────────────────────────────

nodes = []
edges = []


def add_node(nid, label, props):
    props["id"] = nid
    props["label"] = label
    nodes.append(props)


def add_edge(source, target, rel_type, props=None):
    e = {"source": source, "target": target, "relationship": rel_type}
    if props:
        e.update(props)
    edges.append(e)


# --- Law ---
law_metadata = {
    "title_th": "พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562",
    "title_en": "Personal Data Protection Act B.E. 2562 (2019)",
    "abbreviation": "PDPA",
    "gazette_date": "2562-05-27",
    "effective_date": "2565-06-01",
    "total_sections": 96,
    "source": "สำนักงานคณะกรรมการกฤษฎีกา",
    "source_url": "https://searchlaw.ocs.go.th",
}
add_node("PDPA_2562", "Law", law_metadata)

# --- Chapters ---
chapters = [
    {"id": "ch0", "number": 0, "title_th": "บททั่วไป (มาตรา 1-7)", "title_en": "General Provisions", "section_start": 1, "section_end": 7},
    {"id": "ch1", "number": 1, "title_th": "คณะกรรมการคุ้มครองข้อมูลส่วนบุคคล", "title_en": "Personal Data Protection Committee", "section_start": 8, "section_end": 18},
    {"id": "ch2", "number": 2, "title_th": "การคุ้มครองข้อมูลส่วนบุคคล", "title_en": "Personal Data Protection", "section_start": 19, "section_end": 29},
    {"id": "ch3", "number": 3, "title_th": "สิทธิของเจ้าของข้อมูลส่วนบุคคล", "title_en": "Rights of Data Subject", "section_start": 30, "section_end": 42},
    {"id": "ch4", "number": 4, "title_th": "สำนักงานคณะกรรมการคุ้มครองข้อมูลส่วนบุคคล", "title_en": "Office of PDPC", "section_start": 43, "section_end": 71},
    {"id": "ch5", "number": 5, "title_th": "การร้องเรียน", "title_en": "Complaints", "section_start": 72, "section_end": 76},
    {"id": "ch6", "number": 6, "title_th": "ความรับผิดทางแพ่ง", "title_en": "Civil Liability", "section_start": 77, "section_end": 78},
    {"id": "ch7", "number": 7, "title_th": "บทกำหนดโทษ", "title_en": "Penalties", "section_start": 79, "section_end": 90},
    {"id": "ch_trans", "number": 8, "title_th": "บทเฉพาะกาล", "title_en": "Transitional Provisions", "section_start": 91, "section_end": 96},
]

for ch in chapters:
    add_node(ch["id"], "Chapter", {
        "number": ch["number"],
        "title_th": ch["title_th"],
        "title_en": ch["title_en"],
    })
    add_edge("PDPA_2562", ch["id"], "HAS_CHAPTER", {"order": ch["number"]})

# --- Parts ---
parts = [
    {"id": "ch2_p1", "chapter": "ch2", "number": 1, "title_th": "บททั่วไป", "title_en": "General Provisions", "section_start": 19, "section_end": 21},
    {"id": "ch2_p2", "chapter": "ch2", "number": 2, "title_th": "การเก็บรวบรวมข้อมูลส่วนบุคคล", "title_en": "Collection of Personal Data", "section_start": 22, "section_end": 26},
    {"id": "ch2_p3", "chapter": "ch2", "number": 3, "title_th": "การใช้หรือเปิดเผยข้อมูลส่วนบุคคล", "title_en": "Use or Disclosure of Personal Data", "section_start": 27, "section_end": 29},
    {"id": "ch7_p1", "chapter": "ch7", "number": 1, "title_th": "โทษอาญา", "title_en": "Criminal Penalties", "section_start": 79, "section_end": 81},
    {"id": "ch7_p2", "chapter": "ch7", "number": 2, "title_th": "โทษทางปกครอง", "title_en": "Administrative Penalties", "section_start": 82, "section_end": 90},
]

for p in parts:
    add_node(p["id"], "Part", {
        "number": p["number"],
        "title_th": p["title_th"],
        "title_en": p["title_en"],
    })
    add_edge(p["chapter"], p["id"], "HAS_PART", {"order": p["number"]})

# --- Sections ---
def get_chapter_for_section(sec_num):
    for ch in chapters:
        if ch["section_start"] <= sec_num <= ch["section_end"]:
            return ch["id"]
    return None


def get_part_for_section(sec_num):
    for p in parts:
        if p["section_start"] <= sec_num <= p["section_end"]:
            return p["id"]
    return None


for sec_num in sorted(sections.keys()):
    sec_id = f"sec_{sec_num}"
    add_node(sec_id, "Section", {
        "number": sec_num,
        "text": sections[sec_num][:3000],
    })

    ch = get_chapter_for_section(sec_num)
    if ch:
        add_edge(ch, sec_id, "CONTAINS_SECTION", {"order": sec_num})

    part = get_part_for_section(sec_num)
    if part:
        add_edge(part, sec_id, "CONTAINS_SECTION", {"order": sec_num})

# --- Cross-references ---
for sec_num, refs in cross_refs.items():
    for ref in refs:
        add_edge(f"sec_{sec_num}", f"sec_{ref}", "REFERENCES_SECTION")

# --- Definitions ---
definitions = [
    {"id": "def_personal_data", "term_th": "ข้อมูลส่วนบุคคล", "term_en": "Personal Data",
     "definition": "ข้อมูลเกี่ยวกับบุคคลซึ่งทำให้สามารถระบุตัวบุคคลนั้นได้ ไม่ว่าทางตรงหรือทางอ้อม แต่ไม่รวมถึงข้อมูลของผู้ถึงแก่กรรมโดยเฉพาะ", "section": 6},
    {"id": "def_data_controller", "term_th": "ผู้ควบคุมข้อมูลส่วนบุคคล", "term_en": "Data Controller",
     "definition": "บุคคลหรือนิติบุคคลซึ่งมีอำนาจหน้าที่ตัดสินใจเกี่ยวกับการเก็บรวบรวม ใช้ หรือเปิดเผยข้อมูลส่วนบุคคล", "section": 6},
    {"id": "def_data_processor", "term_th": "ผู้ประมวลผลข้อมูลส่วนบุคคล", "term_en": "Data Processor",
     "definition": "บุคคลหรือนิติบุคคลซึ่งดำเนินการเกี่ยวกับการเก็บรวบรวม ใช้ หรือเปิดเผยข้อมูลส่วนบุคคลตามคำสั่งหรือในนามของผู้ควบคุมข้อมูลส่วนบุคคล", "section": 6},
    {"id": "def_data_subject", "term_th": "เจ้าของข้อมูลส่วนบุคคล", "term_en": "Data Subject",
     "definition": "บุคคลธรรมดาที่ข้อมูลส่วนบุคคลนั้นระบุไปถึง", "section": 6},
    {"id": "def_committee", "term_th": "คณะกรรมการ", "term_en": "Personal Data Protection Committee (PDPC)",
     "definition": "คณะกรรมการคุ้มครองข้อมูลส่วนบุคคล", "section": 6},
    {"id": "def_office", "term_th": "สำนักงาน", "term_en": "Office of PDPC",
     "definition": "สำนักงานคณะกรรมการคุ้มครองข้อมูลส่วนบุคคล", "section": 6},
    {"id": "def_secretary", "term_th": "เลขาธิการ", "term_en": "Secretary-General",
     "definition": "เลขาธิการคณะกรรมการคุ้มครองข้อมูลส่วนบุคคล", "section": 6},
    {"id": "def_dpo", "term_th": "เจ้าหน้าที่คุ้มครองข้อมูลส่วนบุคคล", "term_en": "Data Protection Officer (DPO)",
     "definition": "บุคคลที่ได้รับแต่งตั้งเพื่อให้คำแนะนำ ตรวจสอบการดำเนินงาน และประสานงานเกี่ยวกับการคุ้มครองข้อมูลส่วนบุคคล", "section": 41},
    {"id": "def_sensitive_data", "term_th": "ข้อมูลอ่อนไหว", "term_en": "Sensitive Personal Data",
     "definition": "ข้อมูลเกี่ยวกับเชื้อชาติ เผ่าพันธุ์ ความคิดเห็นทางการเมือง ความเชื่อในลัทธิ ศาสนาหรือปรัชญา พฤติกรรมทางเพศ ประวัติอาชญากรรม ข้อมูลสุขภาพ ความพิการ ข้อมูลสหภาพแรงงาน ข้อมูลพันธุกรรม ข้อมูลชีวภาพ", "section": 26},
    {"id": "def_consent", "term_th": "ความยินยอม", "term_en": "Consent",
     "definition": "การแสดงเจตนาโดยชัดแจ้งของเจ้าของข้อมูลส่วนบุคคลในการอนุญาตให้เก็บรวบรวม ใช้ หรือเปิดเผยข้อมูลส่วนบุคคล", "section": 19},
    {"id": "def_biometric", "term_th": "ข้อมูลชีวภาพ", "term_en": "Biometric Data",
     "definition": "ข้อมูลส่วนบุคคลที่เกิดจากการใช้เทคนิคหรือเทคโนโลยีที่เกี่ยวข้องกับการนำลักษณะเด่นทางกายภาพหรือทางพฤติกรรมของบุคคลมาใช้ทำให้สามารถยืนยันตัวตน", "section": 26},
]

for d in definitions:
    add_node(d["id"], "Definition", {
        "term_th": d["term_th"],
        "term_en": d["term_en"],
        "definition": d["definition"],
    })
    add_edge(f"sec_{d['section']}", d["id"], "DEFINES")

# --- Lawful Bases ---
lawful_bases = [
    {"id": "lb_consent", "name_th": "ฐานความยินยอม", "name_en": "Consent", "section": 19},
    {"id": "lb_archive", "name_th": "ฐานจดหมายเหตุ/วิจัย/สถิติ", "name_en": "Archival/Research/Statistics", "section": 24},
    {"id": "lb_vital", "name_th": "ฐานประโยชน์สำคัญต่อชีวิต", "name_en": "Vital Interest", "section": 24},
    {"id": "lb_contract", "name_th": "ฐานสัญญา", "name_en": "Contract", "section": 24},
    {"id": "lb_public_task", "name_th": "ฐานภารกิจรัฐ", "name_en": "Public Task", "section": 24},
    {"id": "lb_legitimate", "name_th": "ฐานประโยชน์โดยชอบธรรม", "name_en": "Legitimate Interest", "section": 24},
    {"id": "lb_legal", "name_th": "ฐานหน้าที่ตามกฎหมาย", "name_en": "Legal Obligation", "section": 24},
]
for lb in lawful_bases:
    add_node(lb["id"], "LawfulBasis", {"name_th": lb["name_th"], "name_en": lb["name_en"]})
    add_edge(f"sec_{lb['section']}", lb["id"], "ESTABLISHES_LAWFUL_BASIS")

# --- Rights ---
rights = [
    {"id": "right_access", "name_th": "สิทธิในการเข้าถึง", "name_en": "Right of Access", "section": 30},
    {"id": "right_portability", "name_th": "สิทธิในการโอนย้ายข้อมูล", "name_en": "Right to Data Portability", "section": 31},
    {"id": "right_object", "name_th": "สิทธิในการคัดค้าน", "name_en": "Right to Object", "section": 32},
    {"id": "right_erasure", "name_th": "สิทธิในการลบข้อมูล", "name_en": "Right to Erasure", "section": 33},
    {"id": "right_restrict", "name_th": "สิทธิในการระงับการใช้", "name_en": "Right to Restrict Processing", "section": 34},
    {"id": "right_rectification", "name_th": "สิทธิในการแก้ไขข้อมูล", "name_en": "Right to Rectification", "section": 36},
    {"id": "right_withdraw", "name_th": "สิทธิถอนความยินยอม", "name_en": "Right to Withdraw Consent", "section": 19},
    {"id": "right_complaint", "name_th": "สิทธิในการร้องเรียน", "name_en": "Right to Complain", "section": 73},
]
for r in rights:
    add_node(r["id"], "Right", {"name_th": r["name_th"], "name_en": r["name_en"]})
    add_edge(f"sec_{r['section']}", r["id"], "GRANTS_RIGHT")
    add_edge("def_data_subject", r["id"], "HAS_RIGHT")

# --- Obligations ---
obligations = [
    {"id": "obl_security", "name_th": "มาตรการรักษาความมั่นคงปลอดภัย", "section": 37},
    {"id": "obl_breach_notify", "name_th": "แจ้งเหตุละเมิดภายใน 72 ชม.", "section": 37},
    {"id": "obl_dpo", "name_th": "แต่งตั้ง DPO", "section": 41},
    {"id": "obl_ropa", "name_th": "จัดทำบันทึกรายการกิจกรรม (ROPA)", "section": 39},
    {"id": "obl_privacy_notice", "name_th": "แจ้งวัตถุประสงค์ (Privacy Notice)", "section": 23},
    {"id": "obl_cross_border", "name_th": "มาตรฐานการโอนข้อมูลข้ามประเทศ", "section": 28},
    {"id": "obl_data_minimization", "name_th": "เก็บเท่าที่จำเป็น", "section": 22},
]
for ob in obligations:
    add_node(ob["id"], "Obligation", {"name_th": ob["name_th"]})
    add_edge(f"sec_{ob['section']}", ob["id"], "IMPOSES_OBLIGATION")
    add_edge("def_data_controller", ob["id"], "MUST_COMPLY_WITH")

# --- Penalties ---
penalties = [
    {"id": "pen_criminal_disclosure", "type": "อาญา", "section": 79,
     "description": "เปิดเผยข้อมูลส่วนบุคคลโดยมิชอบ: จำคุกไม่เกิน 6 เดือน / ปรับไม่เกิน 500,000 บาท",
     "max_fine": 500000, "max_imprisonment_months": 6},
    {"id": "pen_criminal_profit", "type": "อาญา", "section": 79,
     "description": "เปิดเผยเพื่อแสวงหาประโยชน์มิชอบ: จำคุกไม่เกิน 1 ปี / ปรับไม่เกิน 1,000,000 บาท",
     "max_fine": 1000000, "max_imprisonment_months": 12},
    {"id": "pen_admin_1m", "type": "ปกครอง", "section": 85,
     "description": "ปรับทางปกครองไม่เกิน 1,000,000 บาท", "max_fine": 1000000},
    {"id": "pen_admin_3m", "type": "ปกครอง", "section": 86,
     "description": "ปรับทางปกครองไม่เกิน 3,000,000 บาท", "max_fine": 3000000},
    {"id": "pen_admin_5m", "type": "ปกครอง", "section": 90,
     "description": "ปรับทางปกครองไม่เกิน 5,000,000 บาท", "max_fine": 5000000},
    {"id": "pen_civil", "type": "แพ่ง", "section": 77,
     "description": "ค่าสินไหมทดแทน + ศาลสั่งเพิ่มได้ไม่เกิน 2 เท่า", "max_fine": None},
]
for pen in penalties:
    add_node(pen["id"], "Penalty", {
        "type": pen["type"], "description": pen["description"],
        "max_fine": pen.get("max_fine"), "max_imprisonment_months": pen.get("max_imprisonment_months"),
    })
    add_edge(f"sec_{pen['section']}", pen["id"], "PRESCRIBES_PENALTY")

# --- Exemptions ---
exemptions = [
    {"id": "exempt_personal", "name_th": "กิจกรรมส่วนบุคคล/ครอบครัว", "section": 4},
    {"id": "exempt_security", "name_th": "หน่วยงานรัฐด้านความมั่นคง", "section": 4},
    {"id": "exempt_media", "name_th": "สื่อมวลชน/ศิลปกรรม/วรรณกรรม", "section": 4},
    {"id": "exempt_parliament", "name_th": "สภาผู้แทนราษฎร/วุฒิสภา/รัฐสภา", "section": 4},
    {"id": "exempt_court", "name_th": "การพิจารณาคดีของศาล", "section": 4},
    {"id": "exempt_credit", "name_th": "บริษัทข้อมูลเครดิต", "section": 4},
]
for ex in exemptions:
    add_node(ex["id"], "Exemption", {"name_th": ex["name_th"]})
    add_edge(f"sec_{ex['section']}", ex["id"], "GRANTS_EXEMPTION")

# --- Principles ---
principles = [
    {"id": "prin_lawfulness", "name_th": "ชอบด้วยกฎหมาย เป็นธรรม โปร่งใส", "name_en": "Lawfulness, Fairness, Transparency", "sections": [19, 21, 23]},
    {"id": "prin_purpose", "name_th": "จำกัดวัตถุประสงค์", "name_en": "Purpose Limitation", "sections": [21, 22]},
    {"id": "prin_minimization", "name_th": "เก็บเท่าที่จำเป็น", "name_en": "Data Minimization", "sections": [22]},
    {"id": "prin_accuracy", "name_th": "ความถูกต้อง", "name_en": "Accuracy", "sections": [36]},
    {"id": "prin_storage", "name_th": "จำกัดระยะเวลาจัดเก็บ", "name_en": "Storage Limitation", "sections": [23, 37]},
    {"id": "prin_security", "name_th": "ความมั่นคงปลอดภัย", "name_en": "Integrity & Confidentiality", "sections": [37, 40]},
    {"id": "prin_accountability", "name_th": "ความรับผิดชอบ", "name_en": "Accountability", "sections": [37, 39, 41]},
]
for pr in principles:
    add_node(pr["id"], "Principle", {"name_th": pr["name_th"], "name_en": pr["name_en"]})
    for s in pr["sections"]:
        add_edge(f"sec_{s}", pr["id"], "EMBODIES_PRINCIPLE")

# --- Entity Relationships ---
add_edge("def_data_controller", "def_data_processor", "INSTRUCTS")
add_edge("def_data_processor", "def_data_controller", "ACTS_ON_BEHALF_OF")
add_edge("def_data_subject", "def_data_controller", "PROVIDES_CONSENT_TO")
add_edge("def_data_controller", "def_data_subject", "COLLECTS_DATA_FROM")
add_edge("def_data_controller", "def_dpo", "APPOINTS")
add_edge("def_data_processor", "def_dpo", "APPOINTS")
add_edge("def_dpo", "def_office", "COORDINATES_WITH")
add_edge("def_committee", "def_office", "OVERSEES")
add_edge("def_secretary", "def_office", "HEADS")
add_edge("def_committee", "def_data_controller", "REGULATES")
add_edge("def_committee", "def_data_processor", "REGULATES")
add_edge("def_sensitive_data", "def_personal_data", "IS_SUBTYPE_OF")
add_edge("def_biometric", "def_sensitive_data", "IS_SUBTYPE_OF")
add_edge("def_consent", "def_personal_data", "AUTHORIZES_PROCESSING_OF")

# ──────────────────────────────────────────────────────────────
# 5. OUTPUT
# ──────────────────────────────────────────────────────────────

# JSON (main output — used by LightRAG)
graph_json = {
    "metadata": {
        "description": "Knowledge Graph - พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562 (PDPA)",
        "source": "สำนักงานคณะกรรมการกฤษฎีกา",
        "generated_date": "2026-03-14",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_labels": sorted(set(n["label"] for n in nodes)),
        "edge_types": sorted(set(e["relationship"] for e in edges)),
    },
    "nodes": nodes,
    "edges": edges,
}

with open(os.path.join(OUTPUT_DIR, "pdpa_knowledge_graph.json"), "w", encoding="utf-8") as f:
    json.dump(graph_json, f, ensure_ascii=False, indent=2)

# CSV - nodes
with open(os.path.join(OUTPUT_DIR, "nodes.csv"), "w", encoding="utf-8", newline="") as f:
    all_keys = sorted(set(k for n in nodes for k in n.keys()))
    writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
    writer.writeheader()
    for n in nodes:
        writer.writerow(n)

# CSV - edges
with open(os.path.join(OUTPUT_DIR, "edges.csv"), "w", encoding="utf-8", newline="") as f:
    edge_keys = sorted(set(k for e in edges for k in e.keys()))
    writer = csv.DictWriter(f, fieldnames=edge_keys, extrasaction="ignore")
    writer.writeheader()
    for e in edges:
        writer.writerow(e)

# Cypher for Neo4j
def escape_cypher(s):
    if s is None:
        return "null"
    if isinstance(s, (int, float)):
        return str(s)
    return "'" + str(s).replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ") + "'"


cypher_lines = []
cypher_lines.append("// PDPA Knowledge Graph - Neo4j Import")
cypher_lines.append("// Source: สำนักงานคณะกรรมการกฤษฎีกา")
cypher_lines.append("")

for lbl in sorted(set(n["label"] for n in nodes)):
    cypher_lines.append(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{lbl}) REQUIRE n.id IS UNIQUE;")
cypher_lines.append("")

for n in nodes:
    label = n["label"]
    props = {k: v for k, v in n.items() if k != "label" and v is not None}
    prop_str = ", ".join(f"{k}: {escape_cypher(v)}" for k, v in props.items())
    cypher_lines.append(f"MERGE (:{label} {{{prop_str}}});")

cypher_lines.append("")
for e in edges:
    rel = e["relationship"]
    extra = {k: v for k, v in e.items() if k not in ("source", "target", "relationship") and v is not None}
    prop_str = ""
    if extra:
        prop_str = " {" + ", ".join(f"{k}: {escape_cypher(v)}" for k, v in extra.items()) + "}"
    cypher_lines.append(f"MATCH (a {{id: {escape_cypher(e['source'])}}}), (b {{id: {escape_cypher(e['target'])}}}) MERGE (a)-[:{rel}{prop_str}]->(b);")

with open(os.path.join(OUTPUT_DIR, "pdpa_neo4j_import.cypher"), "w", encoding="utf-8") as f:
    f.write("\n".join(cypher_lines))

# NetworkX JSON
nx_data = {
    "directed": True,
    "multigraph": True,
    "graph": graph_json["metadata"],
    "nodes": [{"id": n["id"], **{k: v for k, v in n.items() if k != "id" and v is not None}} for n in nodes],
    "links": [{"source": e["source"], "target": e["target"], "key": e["relationship"],
               **{k: v for k, v in e.items() if k not in ("source", "target", "relationship") and v is not None}} for e in edges],
}
with open(os.path.join(OUTPUT_DIR, "pdpa_networkx.json"), "w", encoding="utf-8") as f:
    json.dump(nx_data, f, ensure_ascii=False, indent=2)

# Stats
print("\n" + "=" * 60)
print("PDPA Knowledge Graph - Build Complete")
print("=" * 60)
print(f"\nTotal Nodes: {len(nodes)}")
print(f"Total Edges: {len(edges)}")
print("\nNode Types:")
node_counts = {}
for n in nodes:
    node_counts[n["label"]] = node_counts.get(n["label"], 0) + 1
for k, v in sorted(node_counts.items()):
    print(f"  {k}: {v}")
print("\nEdge Types:")
edge_counts = {}
for e in edges:
    edge_counts[e["relationship"]] = edge_counts.get(e["relationship"], 0) + 1
for k, v in sorted(edge_counts.items()):
    print(f"  {k}: {v}")
print(f"\nOutput: {OUTPUT_DIR}/")
