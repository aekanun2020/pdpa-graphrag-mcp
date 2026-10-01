"""
LLM Provider — wrapper สำหรับเรียก LLM (PDPA RAG version)
==========================================================
รองรับ OpenRouter (default: gpt-oss-120b) และ Ollama (local)
"""

import os
import requests
import json

# --- Config จาก env var ---
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("LLM_MODEL", "gpt-oss-120b")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

# --- Determinism config ---
# temperature=0 (greedy) เป็นค่าเริ่มต้น เพื่อให้ keyword extraction / answer generation
# คงที่เมื่อ query เดียวกัน (แก้อาการผลค้นหาแกว่งจาก non-deterministic LLM ที่อยู่หน้า retrieval)
# override ได้ผ่าน env var หากต้องการความหลากหลาย
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))
# seed คงที่เพื่อ reproducibility (best-effort — OpenRouter/provider บางตัวอาจไม่รองรับ)
# ตั้งค่าเป็นตัวเลขใน env var LLM_SEED เพื่อเปิดใช้; ปล่อยว่าง = ไม่ส่ง seed
_seed_env = os.environ.get("LLM_SEED", "").strip()
LLM_SEED = int(_seed_env) if _seed_env else None


def query_llm(prompt: str, system_prompt: str = "", model: str = None) -> str:
    """เรียก LLM ตาม provider ที่ตั้งไว้"""
    if LLM_PROVIDER == "openrouter":
        return _query_openrouter(prompt, system_prompt, model)
    else:
        return _query_ollama(prompt, system_prompt, model)


def query_llm_json(prompt: str, system_prompt: str = "", model: str = None) -> dict:
    """เรียก LLM แล้ว parse JSON response"""
    raw = query_llm(prompt, system_prompt, model)
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        return json.loads(raw.strip())
    except (json.JSONDecodeError, IndexError):
        return {"raw": raw, "error": "JSON parse failed"}


def _query_openrouter(prompt, system_prompt="", model=None):
    model = model or OPENROUTER_MODEL
    if not OPENROUTER_API_KEY:
        return "[ERROR] OPENROUTER_API_KEY not set"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
    }
    if LLM_SEED is not None:
        payload["seed"] = LLM_SEED
    try:
        resp = requests.post(OPENROUTER_URL, json=payload, headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR] {e}"


def _query_ollama(prompt, system_prompt="", model=None):
    model = model or OLLAMA_MODEL
    options = {"temperature": LLM_TEMPERATURE}
    if LLM_SEED is not None:
        options["seed"] = LLM_SEED
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": options,
        }, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        return "[ERROR] ไม่สามารถเชื่อมต่อ Ollama ได้"
    except Exception as e:
        return f"[ERROR] {e}"
