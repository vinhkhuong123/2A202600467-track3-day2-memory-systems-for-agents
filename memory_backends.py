import json
import os
import fakeredis
from typing import List, Dict, Any
from chromadb import PersistentClient
from langchain_openai import OpenAIEmbeddings

class ShortTermMemory:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.buffer = []

    def add_message(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content})
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]
            
    def get_context(self) -> List[Dict]:
        return self.buffer
        
    def clear(self):
        self.buffer = []

class LongTermMemory:
    def __init__(self):
        self.redis = fakeredis.FakeStrictRedis(decode_responses=True)
        self.key_prefix = "user_profile:"
        
    def update_fact(self, key: str, value: str, ttl_seconds: int = 604800):
        """Lưu fact với TTL (mặc định 7 ngày = 604800 giây) để tránh giữ PII vô thời hạn."""
        full_key = f"{self.key_prefix}{key}"
        self.redis.set(full_key, value, ex=ttl_seconds)
        
    def get_profile(self) -> Dict[str, str]:
        keys = self.redis.keys(f"{self.key_prefix}*")
        profile = {}
        for k in keys:
            fact_key = k[len(self.key_prefix):]
            profile[fact_key] = self.redis.get(k)
        return profile
        
    def clear(self):
        self.redis.flushall()

class EpisodicMemory:
    def __init__(self, log_file="episodic.jsonl"):
        self.log_file = log_file
        
    def save_episode(self, episode_data: Dict):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(episode_data, ensure_ascii=False) + "\n")
            
    def get_episodes(self, limit=5) -> List[Dict]:
        if not os.path.exists(self.log_file):
            return []
        episodes = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                episodes.append(json.loads(line.strip()))
        return episodes[-limit:]
        
    def clear(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

class SemanticMemory:
    def __init__(self, persist_dir="./chroma_db"):
        self.client = PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("semantic_knowledge")
        self.embeddings = OpenAIEmbeddings()
        
    def add_knowledge(self, text_chunks: List[str], metadatas: List[Dict] = None):
        if not text_chunks:
            return
        embeds = self.embeddings.embed_documents(text_chunks)
        ids = [f"doc_{i}" for i in range(self.collection.count(), self.collection.count() + len(text_chunks))]
        
        kwargs = {
            "embeddings": embeds,
            "documents": text_chunks,
            "ids": ids
        }
        if metadatas:
            kwargs["metadatas"] = metadatas
            
        self.collection.add(**kwargs)
        
    def search(self, query: str, top_k=2) -> List[str]:
        if self.collection.count() == 0:
            return []
        query_embed = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embed],
            n_results=top_k
        )
        return results["documents"][0] if results["documents"] else []
