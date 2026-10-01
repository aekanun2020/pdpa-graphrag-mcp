FROM python:3.11-slim

WORKDIR /app

# gcc จำเป็นสำหรับ build zlib-state (dependency ของ FlagEmbedding → ir-datasets)
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev zlib1g-dev && rm -rf /var/lib/apt/lists/*

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake BGE-M3 model ใส่ image (~2.3 GB)
RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)"

# Copy โค้ด
COPY mcp_server_rag.py .
COPY lightrag/ ./lightrag/

# Data จะ mount ผ่าน docker-compose volume
# (pdpa_full_text_v2.md, pdpa_knowledge_graph.json)

EXPOSE 8100

CMD ["python", "mcp_server_rag.py"]
