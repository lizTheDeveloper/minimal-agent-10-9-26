"""Everything that touches storage: the SQLite file, embeddings, and the FAISS index built from them."""
import json
import os
import sqlite3

import faiss  # vector index: finds the stored embeddings closest to a query embedding
import mlx.core as mx
import numpy as np
from mlx_lm import load

# Qwen3-Embedding runs locally on Apple Silicon via MLX. Swap in a bigger one with EMBED_MODEL, e.g. the 8B.
EMBED_MODEL = os.getenv("EMBED_MODEL", "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")
embed_model, embed_tokenizer = load(EMBED_MODEL)

memory_database = sqlite3.connect("memory.db")  # SQLite holds the text AND its embedding; FAISS is rebuilt from it
memory_cursor = memory_database.cursor()
memory_cursor.execute("CREATE TABLE IF NOT EXISTS memories (content TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
if "embedding" not in [col[1] for col in memory_cursor.execute("PRAGMA table_info(memories)")]:
    memory_cursor.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")  # upgrade older text-only databases
memory_cursor.execute("CREATE TABLE IF NOT EXISTS conversation_history (conversation_json TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
memory_database.commit()


def embed(text, is_query=False):
    """Turn text into a unit-length vector. Qwen3-Embedding wants an instruction on queries, not on documents."""
    if is_query:
        text = f"Instruct: Given a question, retrieve memories that help answer it\nQuery:{text}"
    token_ids = embed_tokenizer.encode(text)  # the tokenizer appends <|endoftext|>, the token we pool on
    hidden = embed_model.model(mx.array([token_ids]))  # run the transformer, skip the next-token head
    vector = np.array(hidden[0, -1].astype(mx.float32))  # last-token pooling: the final token's hidden state
    return vector / np.linalg.norm(vector)  # normalize so inner product == cosine similarity


def build_index():
    """Load every memory's embedding into FAISS, keyed by its SQLite rowid. Embeds any rows that lack one."""
    for rowid, content in memory_cursor.execute("SELECT rowid, content FROM memories WHERE embedding IS NULL").fetchall():
        memory_cursor.execute("UPDATE memories SET embedding = ? WHERE rowid = ?", (embed(content).tobytes(), rowid))
    memory_database.commit()
    rows = memory_cursor.execute("SELECT rowid, embedding FROM memories").fetchall()
    dimension = embed_model.args.hidden_size  # 1024 for 0.6B, 4096 for 8B
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))  # exact inner-product search; IDs map back to rowids
    if rows:
        vectors = np.stack([np.frombuffer(blob, dtype=np.float32) for _, blob in rows])
        index.add_with_ids(vectors, np.array([rowid for rowid, _ in rows], dtype=np.int64))
    return index


memory_index = build_index()


def store_memory(content):
    """Embed a memory, save it to SQLite, and add it to the live FAISS index."""
    vector = embed(content)
    memory_cursor.execute("INSERT INTO memories (content, embedding) VALUES (?, ?)", (content, vector.tobytes()))
    memory_database.commit()
    memory_index.add_with_ids(vector[None, :], np.array([memory_cursor.lastrowid], dtype=np.int64))


def find_memories(query, k=5):
    """Return (content, created_at, similarity) for the k memories closest in meaning to the query."""
    if memory_index.ntotal == 0:
        return []
    scores, rowids = memory_index.search(embed(query, is_query=True)[None, :], min(k, memory_index.ntotal))
    results = []
    for score, rowid in zip(scores[0], rowids[0]):
        content, created_at = memory_cursor.execute("SELECT content, created_at FROM memories WHERE rowid = ?", (int(rowid),)).fetchone()
        results.append((content, created_at, float(score)))
    return results


def save_conversation_history(history):
    """Overwrite the saved conversation with the latest one, so the table always holds where we left off."""
    memory_cursor.execute("DELETE FROM conversation_history")  # throw away the old snapshot
    memory_cursor.execute("INSERT INTO conversation_history (conversation_json) VALUES (?)", (json.dumps(history),))
    memory_database.commit()


def find_in_conversation_history(query):
    """Return the saved messages whose text contains the query (case-insensitive)."""
    row = memory_cursor.execute("SELECT conversation_json FROM conversation_history").fetchone()
    history = json.loads(row[0]) if row else []
    return [message for message in history if query.lower() in (message.get("content") or "").lower()]
