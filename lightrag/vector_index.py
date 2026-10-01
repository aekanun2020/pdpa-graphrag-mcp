"""
Vector Index — BGE-M3 (Dense + Sparse) + Qdrant (PDPA RAG version)
==================================================================
ปรับจาก XXX version — ตัด fact graph profiles, เหลือ knowledge + chunks
"""

import os
import uuid
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, SparseVectorParams, SparseIndexParams,
    Distance, PointStruct, SparseVector,
    Filter, FieldCondition, MatchValue,
    Prefetch, FusionQuery, Fusion,
)

PROFILE_COLLECTION = "pdpa_profiles"
CHUNK_COLLECTION = "pdpa_chunks"
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))


class VectorIndex:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        print(f"  Loading BGE-M3 model...")
        self.model = BGEM3FlagModel(model_name, use_fp16=True)
        self.dim = 1024
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.profiles = []
        self.chunk_store = {}

    def _embed(self, texts: list[str]) -> dict:
        result = self.model.encode(
            texts, batch_size=32, max_length=512,
            return_dense=True, return_sparse=True,
        )
        return {"dense": result["dense_vecs"], "sparse": result["lexical_weights"]}

    @staticmethod
    def _to_sparse_vector(lexical_weights: dict) -> SparseVector:
        indices, values = [], []
        for token_id, weight in lexical_weights.items():
            indices.append(int(token_id))
            values.append(float(weight))
        return SparseVector(indices=indices, values=values)

    @staticmethod
    def _keys_to_text(profile: dict) -> str:
        keys = profile.get("keys", [])
        return " ".join(str(k) for k in keys if k and str(k).strip())

    # ─── Profile Index ────────────────────────────────────────

    def build_profiles(self, profiles: list[dict]):
        """embed knowledge graph profiles แล้ว upsert เข้า Qdrant"""
        self.profiles = profiles

        if self.client.collection_exists(PROFILE_COLLECTION):
            self.client.delete_collection(PROFILE_COLLECTION)

        self.client.create_collection(
            collection_name=PROFILE_COLLECTION,
            vectors_config={
                "dense": VectorParams(size=self.dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            },
        )

        texts = [self._keys_to_text(p) for p in profiles]
        print(f"  Embedding {len(texts)} profiles (keys only)...")
        embeddings = self._embed(texts)

        points = []
        for i, p in enumerate(profiles):
            dense_vec = embeddings["dense"][i]
            sparse_vec = self._to_sparse_vector(embeddings["sparse"][i])
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": dense_vec.tolist(), "sparse": sparse_vec},
                payload={
                    "index": i,
                    "profile_id": p.get("id", ""),
                    "node_category": p.get("node_category", ""),
                },
            ))

        batch_size = 500
        for start in range(0, len(points), batch_size):
            batch = points[start:start + batch_size]
            self.client.upsert(collection_name=PROFILE_COLLECTION, points=batch)

        print(f"  ✅ Upserted {len(points)} profile vectors to Qdrant ({QDRANT_HOST}:{QDRANT_PORT})")

    def search_profiles(self, query: str, top_k: int = 10) -> list[dict]:
        embeddings = self._embed([query])
        dense_vec = embeddings["dense"][0]
        sparse_vec = self._to_sparse_vector(embeddings["sparse"][0])

        results = self.client.query_points(
            collection_name=PROFILE_COLLECTION,
            prefetch=[
                Prefetch(query=dense_vec.tolist(), using="dense", limit=top_k * 2),
                Prefetch(query=sparse_vec, using="sparse", limit=top_k * 2),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        )

        output = []
        for point in results.points:
            idx = point.payload.get("index", 0)
            if idx < len(self.profiles):
                output.append({"profile": self.profiles[idx], "score": point.score})
        return output

    # ─── Chunk Index ──────────────────────────────────────────

    def build_chunks(self, chunk_store: dict):
        """embed มาตรา chunks แล้ว upsert เข้า Qdrant"""
        self.chunk_store = chunk_store

        if not chunk_store:
            print("  ⚠️ chunk_store ว่าง — ข้าม build_chunks")
            return

        if self.client.collection_exists(CHUNK_COLLECTION):
            self.client.delete_collection(CHUNK_COLLECTION)

        self.client.create_collection(
            collection_name=CHUNK_COLLECTION,
            vectors_config={
                "dense": VectorParams(size=self.dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            },
        )

        chunk_ids = sorted(chunk_store.keys())
        texts = [chunk_store[cid]["text"] for cid in chunk_ids]
        print(f"  Embedding {len(texts)} chunks (full text)...")
        embeddings = self._embed(texts)

        points = []
        for i, cid in enumerate(chunk_ids):
            chunk = chunk_store[cid]
            dense_vec = embeddings["dense"][i]
            sparse_vec = self._to_sparse_vector(embeddings["sparse"][i])
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": dense_vec.tolist(), "sparse": sparse_vec},
                payload={
                    "chunk_id": cid,
                    "section_id": chunk.get("section_id", ""),
                    "header_2": chunk.get("header_2", ""),
                    "header_3": chunk.get("header_3", ""),
                },
            ))

        batch_size = 500
        for start in range(0, len(points), batch_size):
            batch = points[start:start + batch_size]
            self.client.upsert(collection_name=CHUNK_COLLECTION, points=batch)

        print(f"  ✅ Upserted {len(points)} chunk vectors to Qdrant '{CHUNK_COLLECTION}'")

    def search_chunks(self, query: str, top_k: int = 5) -> list[dict]:
        """ค้นหาเนื้อหามาตรา PDPA จาก Qdrant"""
        if not self.client.collection_exists(CHUNK_COLLECTION):
            return []

        embeddings = self._embed([query])
        dense_vec = embeddings["dense"][0]
        sparse_vec = self._to_sparse_vector(embeddings["sparse"][0])

        results = self.client.query_points(
            collection_name=CHUNK_COLLECTION,
            prefetch=[
                Prefetch(query=dense_vec.tolist(), using="dense", limit=top_k * 2),
                Prefetch(query=sparse_vec, using="sparse", limit=top_k * 2),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        )

        output = []
        for point in results.points:
            cid = point.payload.get("chunk_id", "")
            chunk_data = self.chunk_store.get(cid, {})
            output.append({
                "chunk_id": cid,
                "text": chunk_data.get("text", ""),
                "section_id": point.payload.get("section_id", ""),
                "header_2": point.payload.get("header_2", ""),
                "header_3": point.payload.get("header_3", ""),
                "score": point.score,
            })
        return output
