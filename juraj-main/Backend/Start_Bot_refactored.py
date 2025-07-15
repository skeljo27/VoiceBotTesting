import os
import numpy as np
import json
import pickle
import requests
from sentence_transformers import SentenceTransformer
import faiss

SIMILARITY_THRESHOLD = 0.7
MAX_CONTEXT_CHARS = 1000
MAX_TOKENS = 300
LLM_ENDPOINT = "http://localhost:1234/v1/chat/completions"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str) -> np.ndarray:
    return embedding_model.encode(text)

with open("data/intents.json", "r", encoding="utf-8") as f:
    all_intents = json.load(f)

with open("data/docs_raw.pkl", "rb") as f:
    docs_raw = pickle.load(f)

with open("data/docs.pkl", "rb") as f:
    docs = pickle.load(f)

index = faiss.read_index("data/karlsruhe_index.faiss")

def detect_intent(user_input: str, all_intents: list[str]) -> str | None:
    best_score = 0.0
    best_intent = None
    user_vec = embed(user_input)
    for intent in all_intents:
        intent_vec = embed(intent)
        sim = np.dot(user_vec, intent_vec) / (np.linalg.norm(user_vec) * np.linalg.norm(intent_vec))
        if sim > best_score:
            best_score = sim
            best_intent = intent
    return best_intent if best_score > 0.5 else None

def generate_response(user_input: str) -> str:
    try:
        vector = embed(user_input)
        intent = detect_intent(user_input, all_intents)

        matched_chunks = []
        if intent:
            filtered_docs = [d for d in docs_raw if d.get("metadata", {}).get("intent") == intent]
            for d in filtered_docs:
                vec = embed(d["page_content"])
                sim = np.dot(vector, vec) / (np.linalg.norm(vector) * np.linalg.norm(vec))
                if sim > SIMILARITY_THRESHOLD:
                    matched_chunks.append(d["page_content"][:MAX_CONTEXT_CHARS])

        if not matched_chunks:
            D, I = index.search(np.array([vector]), k=3)
            for rank, idx in enumerate(I[0]):
                if D[0][rank] > SIMILARITY_THRESHOLD and idx < len(docs):
                    doc = docs[idx]
                    text = doc[:MAX_CONTEXT_CHARS] if isinstance(doc, str) else doc.get("page_content", "")
                    matched_chunks.append(text.strip())

        context = "\n\n".join(matched_chunks[:3])
        prompt_text = (
            "You are a helpful voice assistant for municipal services in Karlsruhe.\n"
            "Speak clearly and simply. Give short and complete answers in 1 to 3 sentences.\n"
            "Avoid unnecessary detail or legal language.\n\n"
            f"Question: {user_input}\n\n"
            f"Context:\n{context}"
        )

        payload = {
            "model": "mistral-7b-instruct-v0.1.Q4_K_M",
            "messages": [{"role": "user", "content": prompt_text.strip()}],
            "stream": False,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.3
        }

        response = requests.post(LLM_ENDPOINT, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"[RAG Error] {e}"
