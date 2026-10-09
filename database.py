"""Everything that touches storage: the SQLite file, embeddings, and the FAISS index built from them."""
import json
import os
import re
import sqlite3

import faiss  # vector index: finds the stored embeddings closest to a query embedding
import numpy as np

try:  # MLX only exists on Apple Silicon Macs
    import mlx.core as mx
    from mlx_lm import load
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# "mlx" runs Qwen3-Embedding locally on Apple Silicon. "openrouter" calls Qwen3-Embedding over the API (any OS).
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "mlx" if HAS_MLX else "openrouter")
if EMBED_BACKEND == "mlx":
    EMBED_MODEL = os.getenv("EMBED_MODEL", "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")
    embed_model, embed_tokenizer = load(EMBED_MODEL)
else:
    from openai import OpenAI
    EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen/qwen3-embedding-8b")
    embed_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

memory_database = sqlite3.connect("memory.db")  # SQLite holds the text AND its embedding; FAISS is rebuilt from it
memory_cursor = memory_database.cursor()
memory_cursor.execute("CREATE TABLE IF NOT EXISTS memories (content TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
if "embedding" not in [col[1] for col in memory_cursor.execute("PRAGMA table_info(memories)")]:
    memory_cursor.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")  # upgrade older text-only databases
memory_cursor.execute("""CREATE TABLE IF NOT EXISTS documents (
    url TEXT UNIQUE, title TEXT, content TEXT, source TEXT, query TEXT, embedding BLOB,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")  # everything a search tool has fetched, so it can be re-searched
memory_cursor.execute("CREATE TABLE IF NOT EXISTS conversation_history (conversation_json TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
memory_database.commit()


MEMORY_TASK = "Given a question, retrieve memories that help answer it"
DOCUMENT_TASK = "Given a question, retrieve web pages and papers that help answer it"
EMBED_CHARS = 4000  # embed only the start of long documents, to keep embedding fast


def embed(text, task=None):
    """Turn text into a unit-length vector. Qwen3-Embedding wants an instruction on queries (task), not on documents."""
    if task:
        text = f"Instruct: {task}\nQuery:{text}"
    if EMBED_BACKEND == "mlx":
        token_ids = embed_tokenizer.encode(text)  # the tokenizer appends <|endoftext|>, the token we pool on
        hidden = embed_model.model(mx.array([token_ids]))  # run the transformer, skip the next-token head
        vector = np.array(hidden[0, -1].astype(mx.float32))  # last-token pooling: the final token's hidden state
    else:
        response = embed_client.embeddings.create(model=EMBED_MODEL, input=text)
        vector = np.array(response.data[0].embedding, dtype=np.float32)
    return vector / np.linalg.norm(vector)  # normalize so inner product == cosine similarity


def build_index(table, dimension):
    """Load every row's embedding from a table into FAISS, keyed by its SQLite rowid."""
    rows = memory_cursor.execute(f"SELECT rowid, embedding FROM {table}").fetchall()
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))  # exact inner-product search; IDs map back to rowids
    if rows:
        vectors = np.stack([np.frombuffer(blob, dtype=np.float32) for _, blob in rows])
        if vectors.shape[1] != dimension:
            raise SystemExit(f"memory.db was built with a different embedding model ({vectors.shape[1]} dims, "
                             f"{EMBED_MODEL} makes {dimension}). Delete memory.db or switch EMBED_BACKEND back.")
        index.add_with_ids(vectors, np.array([rowid for rowid, _ in rows], dtype=np.int64))
    return index


for rowid, content in memory_cursor.execute("SELECT rowid, content FROM memories WHERE embedding IS NULL").fetchall():
    memory_cursor.execute("UPDATE memories SET embedding = ? WHERE rowid = ?", (embed(content).tobytes(), rowid))  # upgrade text-only memories
memory_database.commit()
embedding_dimension = len(embed("dimension check"))  # 1024 for 0.6B, 4096 for 8B
memory_index = build_index("memories", embedding_dimension)
document_index = build_index("documents", embedding_dimension)  # kept apart so web pages don't crowd out memories


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
    scores, rowids = memory_index.search(embed(query, task=MEMORY_TASK)[None, :], min(k, memory_index.ntotal))
    results = []
    for score, rowid in zip(scores[0], rowids[0]):
        content, created_at = memory_cursor.execute("SELECT content, created_at FROM memories WHERE rowid = ?", (int(rowid),)).fetchone()
        results.append((content, created_at, float(score)))
    return results


def clean_markdown(text):
    """Light cleanup for fetched markdown: keep link text but drop the URLs, and collapse whitespace."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [text](url) -> text, and drop image URLs
    text = text.replace("\u200b", "")  # zero-width spaces left behind by "direct link" anchors
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def store_document(url, title, content, source, query):
    """Save a fetched search result and embed it. Re-fetching a URL updates it; unchanged content is skipped."""
    content = clean_markdown(content)
    existing = memory_cursor.execute("SELECT rowid, content FROM documents WHERE url = ?", (url,)).fetchone()
    if existing and existing[1] == content:
        return
    vector = embed(f"{title}\n{content}"[:EMBED_CHARS])
    if existing:
        memory_cursor.execute("UPDATE documents SET title = ?, content = ?, source = ?, query = ?, embedding = ?, "
                              "fetched_at = CURRENT_TIMESTAMP WHERE rowid = ?", (title, content, source, query, vector.tobytes(), existing[0]))
        rowid = existing[0]
        document_index.remove_ids(np.array([rowid], dtype=np.int64))  # drop the old vector before adding the new one
    else:
        memory_cursor.execute("INSERT INTO documents (url, title, content, source, query, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                              (url, title, content, source, query, vector.tobytes()))
        rowid = memory_cursor.lastrowid
    memory_database.commit()
    document_index.add_with_ids(vector[None, :], np.array([rowid], dtype=np.int64))


def find_documents(query, k=5):
    """Return (title, url, content, source, fetched_at, similarity) for the k cached documents closest to the query."""
    if document_index.ntotal == 0:
        return []
    scores, rowids = document_index.search(embed(query, task=DOCUMENT_TASK)[None, :], min(k, document_index.ntotal))
    results = []
    for score, rowid in zip(scores[0], rowids[0]):
        row = memory_cursor.execute("SELECT title, url, content, source, fetched_at FROM documents WHERE rowid = ?", (int(rowid),)).fetchone()
        results.append((*row, float(score)))
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
