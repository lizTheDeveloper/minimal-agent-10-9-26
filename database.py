"""Everything that touches storage: the SQLite file, embeddings, and the FAISS index built from them."""
import html
import json
import os
import re
import sqlite3

# faiss-cpu and torch (for Prompt Guard) each bundle their own OpenMP runtime, which crashes on macOS when both load.
# Allowing the duplicate is only safe single-threaded (multi-threaded it segfaults), and these models are small enough not to need more.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
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

# SQLite holds the text AND its embedding; FAISS is rebuilt from it. The timeout lets a scheduled research run and
# a chat session share the file: a write waits up to 30 seconds for the other one's to finish.
memory_database = sqlite3.connect("memory.db", timeout=30)
memory_cursor = memory_database.cursor()
memory_cursor.execute("CREATE TABLE IF NOT EXISTS memories (content TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
if "embedding" not in [col[1] for col in memory_cursor.execute("PRAGMA table_info(memories)")]:
    memory_cursor.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")  # upgrade older text-only databases
if "tier" not in [col[1] for col in memory_cursor.execute("PRAGMA table_info(memories)")]:
    memory_cursor.execute("ALTER TABLE memories ADD COLUMN tier TEXT DEFAULT 'short_term'")  # see TIERS below
memory_cursor.execute("""CREATE TABLE IF NOT EXISTS documents (
    url TEXT UNIQUE, title TEXT, content TEXT, source TEXT, query TEXT, embedding BLOB,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")  # everything a search tool has fetched, so it can be re-searched
for column in ("thread_id INTEGER", "thread_status TEXT"):  # thread_status: auto, model, or unsure (waiting for the model)
    if column.split()[0] not in [col[1] for col in memory_cursor.execute("PRAGMA table_info(documents)")]:
        memory_cursor.execute(f"ALTER TABLE documents ADD COLUMN {column}")
memory_cursor.execute("""CREATE TABLE IF NOT EXISTS threads (
    id INTEGER PRIMARY KEY, title TEXT, lead_rowid INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")  # news stories
memory_cursor.execute("CREATE TABLE IF NOT EXISTS categories (id INTEGER PRIMARY KEY, name TEXT UNIQUE, description TEXT, embedding BLOB)")
memory_cursor.execute("""CREATE TABLE IF NOT EXISTS thread_categories (
    thread_id INTEGER, category_id INTEGER, status TEXT, PRIMARY KEY (thread_id, category_id))""")  # a story can be in several
memory_cursor.execute("CREATE TABLE IF NOT EXISTS reports (report TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")  # autonomous runs
memory_cursor.execute("CREATE TABLE IF NOT EXISTS conversation_history (conversation_json TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
memory_cursor.execute("""CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY, name TEXT COLLATE NOCASE, type TEXT COLLATE NOCASE DEFAULT '', data TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, UNIQUE (name, type))""")  # knowledge graph nodes: any noun, any JSON
memory_cursor.execute("""CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY, source_id INTEGER, relation TEXT, target_id INTEGER, data TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, UNIQUE (source_id, relation, target_id))""")  # edges: source -relation-> target
if "embedding" not in [col[1] for col in memory_cursor.execute("PRAGMA table_info(entities)")]:
    memory_cursor.execute("ALTER TABLE entities ADD COLUMN embedding BLOB")  # upgrade graphs saved before entities were embedded
memory_database.commit()


MEMORY_TASK = "Given a question, retrieve memories that help answer it"
DOCUMENT_TASK = "Given a question, retrieve web pages and papers that help answer it"
ENTITY_TASK = "Given a question, retrieve knowledge graph entities (people, organizations, things, ideas) it is about"
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


def entity_text(name, type, data_json):
    """What gets embedded for an entity: its name, its type and its data, like 'Ada Lovelace (person): {"born": 1815}'."""
    return f"{name}" + (f" ({type})" if type else "") + (f": {data_json}" if data_json not in (None, "", "{}") else "")


for rowid, name, type, data in memory_cursor.execute("SELECT id, name, type, data FROM entities WHERE embedding IS NULL").fetchall():
    memory_cursor.execute("UPDATE entities SET embedding = ? WHERE id = ?", (embed(entity_text(name, type, data)).tobytes(), rowid))
memory_database.commit()
embedding_dimension = len(embed("dimension check"))  # 1024 for 0.6B, 4096 for 8B
memory_index = build_index("memories", embedding_dimension)
document_index = build_index("documents", embedding_dimension)  # kept apart so web pages don't crowd out memories
entity_index = build_index("entities", embedding_dimension)  # knowledge graph nodes, found by meaning as well as by name
seen_version = memory_cursor.execute("PRAGMA data_version").fetchone()[0]  # changes when ANOTHER process commits


def refresh_indexes():
    """Each running agent keeps its own FAISS indexes in memory. If another process (like a scheduled research run)
    has written to memory.db since we last looked, rebuild them so its new memories, documents and entities show up."""
    global seen_version, memory_index, document_index, entity_index
    version = memory_cursor.execute("PRAGMA data_version").fetchone()[0]
    if version != seen_version:
        seen_version = version
        memory_index = build_index("memories", embedding_dimension)
        document_index = build_index("documents", embedding_dimension)
        entity_index = build_index("entities", embedding_dimension)


def normalize(text):
    """Lowercase, collapse whitespace and drop final punctuation, so trivially different copies compare equal."""
    return " ".join(text.lower().split()).rstrip(".!?")


def contains(longer, shorter):
    """True if longer is a fuller version of shorter: shorter appears in it as whole words (so "cat" isn't found inside
    "concatenate"), and longer is at most three times as long. A much longer memory, like a conversation summary, is a
    different memory that happens to mention the fact, so both are kept."""
    if len(longer) > 3 * len(shorter):
        return False
    return re.search(rf"(?<!\w){re.escape(shorter)}(?!\w)", longer) is not None


# Memory tiers. Core and short-term memories go into every prompt; long-term ones are only found with search_memory.
TIERS = ("core", "short_term", "long_term")  # most to least important
CORE_LIMIT = 20  # core memories are in every prompt, so keep them few
SHORT_TERM_LIMIT = 20  # keep at most this many short-term memories...
SHORT_TERM_DAYS = 3  # ...and none older than this; the rest are pruned down to long-term


def store_memory(content, tier="short_term"):
    """Embed a memory and save it, keeping only the most complete version of anything already remembered.

    Returns "empty", "duplicate" if an existing memory already contains it, "core full", or the number of shorter
    memories it replaced. A memory that replaces others keeps the most important tier among them."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    new = normalize(content)
    if not new:
        return "empty"
    existing = memory_cursor.execute("SELECT rowid, content, tier FROM memories").fetchall()
    if any(contains(normalize(old), new) for _, old, _ in existing):
        return "duplicate"  # exact repeat, or part of something longer we already have
    covered = [(rowid, old_tier) for rowid, old, old_tier in existing if contains(new, normalize(old))]  # shorter memories this one covers
    replaced = [rowid for rowid, _ in covered]
    tier = min([tier] + [old_tier for _, old_tier in covered], key=TIERS.index)  # don't demote a core memory by extending it
    if tier == "core" and "core" not in [old_tier for _, old_tier in covered] and count_tier("core") >= CORE_LIMIT:
        return "core full"
    vector = embed(content)  # before deleting anything, so a failed embedding can't lose the old memories
    if replaced:
        memory_cursor.executemany("DELETE FROM memories WHERE rowid = ?", [(rowid,) for rowid in replaced])
        memory_index.remove_ids(np.array(replaced, dtype=np.int64))
    memory_cursor.execute("INSERT INTO memories (content, embedding, tier) VALUES (?, ?, ?)", (content, vector.tobytes(), tier))
    memory_database.commit()
    memory_index.add_with_ids(vector[None, :], np.array([memory_cursor.lastrowid], dtype=np.int64))
    prune_memories()
    return len(replaced)


def count_tier(tier):
    return memory_cursor.execute("SELECT count(*) FROM memories WHERE tier = ?", (tier,)).fetchone()[0]


def prune_memories():
    """Move short-term memories that are too old, or beyond the newest SHORT_TERM_LIMIT, down to long-term."""
    memory_cursor.execute("UPDATE memories SET tier = 'long_term' WHERE tier = 'short_term' AND "
                          "(created_at < datetime('now', ?) OR rowid NOT IN "
                          "(SELECT rowid FROM memories WHERE tier = 'short_term' ORDER BY created_at DESC, rowid DESC LIMIT ?))",
                          (f"-{SHORT_TERM_DAYS} days", SHORT_TERM_LIMIT))
    memory_database.commit()
    return memory_cursor.rowcount


def dedupe_memories():
    """Remove memories saved before deduplication existed: anything identical to, or contained in, a longer memory.
    The longest version is kept, with the most important tier among the copies."""
    rows = memory_cursor.execute("SELECT rowid, content, tier FROM memories ORDER BY length(content) DESC, rowid").fetchall()
    kept, removed = [], []  # kept: [rowid, normalized content, tier]
    for rowid, content, tier in rows:
        new = normalize(content)
        keeper = next((k for k in kept if contains(k[1], new)), None)
        if keeper is None:
            kept.append([rowid, new, tier])
            continue
        removed.append(rowid)
        if TIERS.index(tier) < TIERS.index(keeper[2]):  # a core copy makes the kept version core
            keeper[2] = tier
            memory_cursor.execute("UPDATE memories SET tier = ? WHERE rowid = ?", (tier, keeper[0]))
    if removed:
        memory_cursor.executemany("DELETE FROM memories WHERE rowid = ?", [(rowid,) for rowid in removed])
        memory_index.remove_ids(np.array(removed, dtype=np.int64))
    memory_database.commit()
    return len(removed)


def memories_in_tier(tier):
    """Return (id, content, created_at) for every memory in a tier, oldest first. Used to build the prompt."""
    return memory_cursor.execute("SELECT rowid, content, created_at FROM memories WHERE tier = ? ORDER BY created_at, rowid", (tier,)).fetchall()


def set_tier(memory_id, tier):
    """Move a memory to another tier. Returns "core full" instead if core has no room."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    row = memory_cursor.execute("SELECT tier FROM memories WHERE rowid = ?", (memory_id,)).fetchone()
    if row is None:
        raise ValueError(f"No memory {memory_id}")
    if tier == "core" and row[0] != "core" and count_tier("core") >= CORE_LIMIT:
        return "core full"
    # A memory moved into short-term counts as new again, so pruning doesn't immediately push it back out.
    timestamp = ", created_at = CURRENT_TIMESTAMP" if tier == "short_term" and row[0] != "short_term" else ""
    memory_cursor.execute(f"UPDATE memories SET tier = ?{timestamp} WHERE rowid = ?", (tier, memory_id))
    memory_database.commit()
    prune_memories()
    return tier


def forget_memory(memory_id):
    """Delete a memory for good."""
    if memory_cursor.execute("DELETE FROM memories WHERE rowid = ?", (memory_id,)).rowcount == 0:
        raise ValueError(f"No memory {memory_id}")
    memory_database.commit()
    memory_index.remove_ids(np.array([memory_id], dtype=np.int64))


def find_memories(query, k=5):
    """Return (id, content, created_at, similarity) for the k long-term memories closest in meaning to the query.
    Core and short-term memories are already in the prompt, so they aren't searched."""
    refresh_indexes()
    if memory_index.ntotal == 0:
        return []
    scores, rowids = memory_index.search(embed(query, task=MEMORY_TASK)[None, :], memory_index.ntotal)  # all, then keep long-term
    results = []
    for score, rowid in zip(scores[0], rowids[0]):
        row = memory_cursor.execute("SELECT content, created_at FROM memories WHERE rowid = ? AND tier = 'long_term'", (int(rowid),)).fetchone()
        if row:
            results.append((int(rowid), *row, float(score)))
        if len(results) == k:
            break
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
    title = html.unescape(title)  # titles often arrive as "SpaceX&#x27;s"
    existing = memory_cursor.execute("SELECT rowid, content FROM documents WHERE url = ?", (url,)).fetchone()
    if existing and existing[1] == content:
        return None
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
    return None if existing else auto_thread(rowid, vector, title)  # re-fetched documents keep their thread


def find_documents(query, k=5):
    """Return (title, url, content, source, fetched_at, similarity) for the k cached documents closest to the query."""
    refresh_indexes()
    if document_index.ntotal == 0:
        return []
    scores, rowids = document_index.search(embed(query, task=DOCUMENT_TASK)[None, :], min(k, document_index.ntotal))
    results = []
    for score, rowid in zip(scores[0], rowids[0]):
        row = memory_cursor.execute("SELECT title, url, content, source, fetched_at FROM documents WHERE rowid = ?", (int(rowid),)).fetchone()
        results.append((*row, float(score)))
    return results


# News threads, memeorandum-style: each thread is one story, with a lead document and the other coverage of it.
# Embeddings decide the clear cases. In between the two thresholds they can't tell, so the language model decides.
SAME_STORY = 0.6  # at or above this similarity to a thread's lead: join that thread automatically
NEW_STORY = 0.45  # below this for every thread: start a new thread automatically


def thread_candidates(vector, k=3):
    """Return (similarity, thread_id, title) for the k threads whose lead document is closest to this vector."""
    leads = memory_cursor.execute("SELECT threads.id, threads.title, documents.embedding FROM threads "
                                  "JOIN documents ON documents.rowid = threads.lead_rowid").fetchall()
    scored = [(float(vector @ np.frombuffer(blob, dtype=np.float32)), thread_id, title) for thread_id, title, blob in leads]
    return sorted(scored, reverse=True)[:k]


def create_thread(rowid, title, status):
    """Start a new thread with this document as its lead."""
    memory_cursor.execute("INSERT INTO threads (title, lead_rowid) VALUES (?, ?)", (title, rowid))
    thread_id = memory_cursor.lastrowid
    memory_cursor.execute("UPDATE documents SET thread_id = ?, thread_status = ? WHERE rowid = ?", (thread_id, status, rowid))
    memory_database.commit()
    auto_categorize(thread_id)


def auto_thread(rowid, vector, title):
    """Put a new document on a thread if the embeddings are sure. Returns "auto" or "unsure"."""
    candidates = thread_candidates(vector, k=1)
    best = candidates[0][0] if candidates else 0.0
    if best >= SAME_STORY:
        memory_cursor.execute("UPDATE documents SET thread_id = ?, thread_status = 'auto' WHERE rowid = ?", (candidates[0][1], rowid))
        memory_database.commit()
        return "auto"
    if best < NEW_STORY:
        create_thread(rowid, title, "auto")
        return "auto"
    memory_cursor.execute("UPDATE documents SET thread_id = NULL, thread_status = 'unsure' WHERE rowid = ?", (rowid,))
    memory_database.commit()
    return "unsure"


def tidy_thread(thread_id):
    """After a document leaves a thread: delete the thread if it's empty, or pick a new lead if the lead left."""
    if thread_id is None:
        return
    members = [row[0] for row in memory_cursor.execute("SELECT rowid FROM documents WHERE thread_id = ? ORDER BY rowid", (thread_id,))]
    if not members:
        memory_cursor.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        memory_cursor.execute("DELETE FROM thread_categories WHERE thread_id = ?", (thread_id,))
    else:
        lead = memory_cursor.execute("SELECT lead_rowid FROM threads WHERE id = ?", (thread_id,)).fetchone()[0]
        if lead not in members:
            memory_cursor.execute("UPDATE threads SET lead_rowid = ? WHERE id = ?", (members[0], thread_id))
    memory_database.commit()


def document_by_url(url):
    """Return (rowid, title, thread_id) for a cached document, or raise if we've never fetched it."""
    row = memory_cursor.execute("SELECT rowid, title, thread_id FROM documents WHERE url = ?", (url,)).fetchone()
    if row is None:
        raise ValueError(f"No cached document with url {url}")
    return row


def move_to_thread(url, thread_id):
    """Put a document on an existing thread (also moves it off its old one)."""
    rowid, _, old_thread = document_by_url(url)
    if memory_cursor.execute("SELECT 1 FROM threads WHERE id = ?", (thread_id,)).fetchone() is None:
        raise ValueError(f"No thread {thread_id}")
    memory_cursor.execute("UPDATE documents SET thread_id = ?, thread_status = 'model' WHERE rowid = ?", (thread_id, rowid))
    tidy_thread(old_thread)


def start_thread(url, title=None):
    """Start a new thread led by this document (also moves it off its old one). Returns the new thread's id."""
    rowid, document_title, old_thread = document_by_url(url)
    create_thread(rowid, title or document_title, "model")
    tidy_thread(old_thread)
    return memory_cursor.execute("SELECT thread_id FROM documents WHERE rowid = ?", (rowid,)).fetchone()[0]


def merge_threads(from_thread_id, into_thread_id):
    """Move every document from one thread into another, and delete the emptied thread."""
    for thread_id in (from_thread_id, into_thread_id):
        if memory_cursor.execute("SELECT 1 FROM threads WHERE id = ?", (thread_id,)).fetchone() is None:
            raise ValueError(f"No thread {thread_id}")
    if from_thread_id == into_thread_id:
        raise ValueError("Can't merge a thread into itself")
    memory_cursor.execute("UPDATE documents SET thread_id = ? WHERE thread_id = ?", (into_thread_id, from_thread_id))
    memory_cursor.execute("INSERT OR IGNORE INTO thread_categories (thread_id, category_id, status) "
                          "SELECT ?, category_id, status FROM thread_categories WHERE thread_id = ?", (into_thread_id, from_thread_id))
    tidy_thread(from_thread_id)


def unsure_documents(limit=10, candidates=3):
    """Return documents the embeddings couldn't place, each as (title, url, content, closest threads)."""
    rows = memory_cursor.execute("SELECT title, url, content, embedding FROM documents WHERE thread_status = 'unsure' "
                                 "ORDER BY rowid LIMIT ?", (limit,)).fetchall()
    return [(title, url, content, thread_candidates(np.frombuffer(blob, dtype=np.float32), candidates))
            for title, url, content, blob in rows]


def thread_lead_text(thread_id):
    """The title and text of a thread's lead document: what the story is about."""
    title, content = memory_cursor.execute("SELECT documents.title, documents.content FROM threads JOIN documents "
                                           "ON documents.rowid = threads.lead_rowid WHERE threads.id = ?", (thread_id,)).fetchone()
    return f"{title}\n{content}"


def list_threads(days=7, category=None):
    """Return threads with documents fetched in the last few days, biggest first, as
    (thread_id, title, [(title, url, content, source) lead first]). Optionally only threads in one category."""
    if category is None:
        thread_rows = memory_cursor.execute("SELECT id, title, lead_rowid FROM threads").fetchall()
    else:
        category_id = category_by_name(category)
        thread_rows = memory_cursor.execute("SELECT id, title, lead_rowid FROM threads JOIN thread_categories ON thread_id = id "
                                            "WHERE category_id = ?", (category_id,)).fetchall()
    threads = []
    for thread_id, title, lead in thread_rows:
        members = memory_cursor.execute("SELECT title, url, content, source FROM documents WHERE thread_id = ? "
                                        "AND fetched_at >= datetime('now', ?) ORDER BY rowid != ?, rowid",
                                        (thread_id, f"-{days} days", lead)).fetchall()
        if members:
            threads.append((thread_id, title, members))
    return sorted(threads, key=lambda thread: len(thread[2]), reverse=True)


# Categories group threads by topic, like "AI security" vs "AI governance". A thread can be in several.
# Same idea as threads: embeddings tag the clear matches, and the model decides the rest.
CATEGORY_TASK = "Given a news category, retrieve news articles that belong in it"
CATEGORY_MATCH = 0.55  # at or above this similarity between a category and a thread's lead: tag it automatically
# (category-vs-document scores run lower than document-vs-document ones and neighbouring topics overlap, so this errs on precision)


def category_by_name(name):
    row = memory_cursor.execute("SELECT id FROM categories WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
    if row is None:
        raise ValueError(f"No category named {name!r}")
    return row[0]


def category_scores(thread_id):
    """Return (similarity, category_id, name) for every category, closest first, against the thread's lead document."""
    lead = memory_cursor.execute("SELECT documents.embedding FROM threads JOIN documents ON documents.rowid = threads.lead_rowid "
                                 "WHERE threads.id = ?", (thread_id,)).fetchone()
    if lead is None:
        return []
    vector = np.frombuffer(lead[0], dtype=np.float32)
    categories = memory_cursor.execute("SELECT id, name, embedding FROM categories").fetchall()
    return sorted(((float(vector @ np.frombuffer(blob, dtype=np.float32)), category_id, name) for category_id, name, blob in categories), reverse=True)


def auto_categorize(thread_id, only_category=None):
    """Tag a thread with every category it clearly matches (or just one, so a new category doesn't undo the model's choices)."""
    for score, category_id, _ in category_scores(thread_id):
        if score >= CATEGORY_MATCH and only_category in (None, category_id):
            memory_cursor.execute("INSERT OR IGNORE INTO thread_categories VALUES (?, ?, 'auto')", (thread_id, category_id))
    memory_database.commit()


def create_category(name, description):
    """Add a category, and tag the existing threads that clearly belong in it. Returns how many were tagged."""
    name = name.strip()
    if memory_cursor.execute("SELECT 1 FROM categories WHERE name = ? COLLATE NOCASE", (name,)).fetchone():  # UNIQUE alone is case-sensitive
        raise ValueError(f"A category named {name!r} already exists")
    vector = embed(f"{name}: {description}", task=CATEGORY_TASK)  # the description is the "query" articles are matched to
    memory_cursor.execute("INSERT INTO categories (name, description, embedding) VALUES (?, ?, ?)", (name, description, vector.tobytes()))
    category_id = memory_cursor.lastrowid
    memory_database.commit()
    for (thread_id,) in memory_cursor.execute("SELECT id FROM threads").fetchall():
        auto_categorize(thread_id, category_id)
    return memory_cursor.execute("SELECT count(*) FROM thread_categories WHERE category_id = ?", (category_id,)).fetchone()[0]


def list_categories():
    """Return (name, description, thread count) for every category."""
    return memory_cursor.execute("SELECT name, description, (SELECT count(*) FROM thread_categories WHERE category_id = id) "
                                 "FROM categories ORDER BY name").fetchall()


def set_thread_categories(thread_id, names):
    """Replace a thread's categories with these (an empty list removes them all)."""
    if memory_cursor.execute("SELECT 1 FROM threads WHERE id = ?", (thread_id,)).fetchone() is None:
        raise ValueError(f"No thread {thread_id}")
    category_ids = [category_by_name(name) for name in names]  # check every name before changing anything
    memory_cursor.execute("DELETE FROM thread_categories WHERE thread_id = ?", (thread_id,))
    memory_cursor.executemany("INSERT OR IGNORE INTO thread_categories VALUES (?, ?, 'model')", [(thread_id, c) for c in category_ids])
    memory_database.commit()


def uncategorized_threads(limit=10, closest=3):
    """Return threads with no category yet, as (thread_id, title, [(similarity, name)] closest categories)."""
    rows = memory_cursor.execute("SELECT id, title FROM threads WHERE id NOT IN (SELECT thread_id FROM thread_categories) "
                                 "ORDER BY id LIMIT ?", (limit,)).fetchall()
    return [(thread_id, title, [(score, name) for score, _, name in category_scores(thread_id)[:closest]]) for thread_id, title in rows]


def thread_category_names(thread_id):
    return [row[0] for row in memory_cursor.execute("SELECT name FROM categories JOIN thread_categories ON category_id = id "
                                                    "WHERE thread_id = ? ORDER BY name", (thread_id,))]


# Knowledge graph: entities (any noun: a person, a company, a paper, an idea) joined by named relationships, like
# "Ada Lovelace -worked_with-> Charles Babbage". Both can carry any JSON data. Tools refer to entities by exact name;
# search_entities finds them by meaning, using an embedding of each entity's name, type and data.


def merge_data(old_json, new_data):
    """Add new_data's keys to stored JSON, overwriting keys that already exist. A key set to None is removed."""
    data = json.loads(old_json or "{}")
    data.update(new_data or {})
    return json.dumps({key: value for key, value in data.items() if value is not None})


def entity_id(name, type=None):
    """Find an entity by name (any case), and by type if given. Raises if there's none, or if the name is ambiguous."""
    name = " ".join(name.split())
    query = "SELECT id, type FROM entities WHERE name = ?" + (" AND type = ?" if type is not None else "")
    rows = memory_cursor.execute(query, (name,) if type is None else (name, type)).fetchall()
    if not rows:
        raise ValueError(f"No entity named {name!r}" + (f" of type {type!r}" if type is not None else ""))
    if len(rows) > 1:
        raise ValueError(f"Several entities are named {name!r} (types: {', '.join(t or 'none' for _, t in rows)}). Say which type.")
    return rows[0][0]


def store_entity(name, type="", data=None):
    """Add an entity, or merge data into it if one with this name and type exists. Returns (id, created)."""
    name, type = " ".join(name.split()), type or ""
    if not name:
        raise ValueError("An entity needs a name")
    row = memory_cursor.execute("SELECT id, data FROM entities WHERE name = ? AND type = ?", (name, type)).fetchone()
    merged = merge_data(row[1] if row else None, data)
    if row and merged == row[1]:
        return row[0], False  # nothing new, so no need to re-embed
    vector = embed(entity_text(name, type, merged))  # the data is embedded too, so new data means a new vector
    if row:
        memory_cursor.execute("UPDATE entities SET data = ?, embedding = ? WHERE id = ?", (merged, vector.tobytes(), row[0]))
        entity_index.remove_ids(np.array([row[0]], dtype=np.int64))
    else:
        memory_cursor.execute("INSERT INTO entities (name, type, data, embedding) VALUES (?, ?, ?, ?)", (name, type, merged, vector.tobytes()))
    memory_database.commit()
    entity = row[0] if row else memory_cursor.lastrowid
    entity_index.add_with_ids(vector[None, :], np.array([entity], dtype=np.int64))
    return entity, row is None


def find_or_create_entity(name, type=None):
    """The entity with this name (and type, if given), creating it if it doesn't exist yet."""
    try:
        return entity_id(name, type)
    except ValueError as error:
        if "Several" in str(error):
            raise
        return store_entity(name, type or "")[0]


def store_relationship(source, relation, target, data=None, source_type=None, target_type=None):
    """Link two entities by name, creating either one if it's new. Merges data into an existing identical link.
    Returns (id, created)."""
    relation = "_".join(relation.lower().split())  # "Works at" and "works_at" are the same relation
    if not relation:
        raise ValueError("A relationship needs a relation, like 'works_at'")
    source_id, target_id = find_or_create_entity(source, source_type), find_or_create_entity(target, target_type)
    row = memory_cursor.execute("SELECT id, data FROM relationships WHERE source_id = ? AND relation = ? AND target_id = ?",
                                (source_id, relation, target_id)).fetchone()
    if row:
        memory_cursor.execute("UPDATE relationships SET data = ? WHERE id = ?", (merge_data(row[1], data), row[0]))
    else:
        memory_cursor.execute("INSERT INTO relationships (source_id, relation, target_id, data) VALUES (?, ?, ?, ?)",
                              (source_id, relation, target_id, merge_data(None, data)))
    memory_database.commit()
    return (row[0], False) if row else (memory_cursor.lastrowid, True)


def entity_row(node):
    """Return (id, name, type, data dict) for an entity."""
    id_, name, type, data = memory_cursor.execute("SELECT id, name, type, data FROM entities WHERE id = ?", (node,)).fetchone()
    return id_, name, type, json.loads(data)


def neighborhood(name, type=None, depth=1):
    """Walk the graph out from one entity, following relationships in both directions, up to depth hops.
    Returns (entities, relationships): entities as (id, name, type, data), relationships as
    (id, source name, relation, target name, data). The starting entity comes first."""
    start = entity_id(name, type)
    seen, frontier, edges = {start}, [start], {}
    for _ in range(depth):
        next_frontier = []
        for node in frontier:
            for edge_id, source_id, relation, target_id, data in memory_cursor.execute(
                    "SELECT id, source_id, relation, target_id, data FROM relationships WHERE source_id = ? OR target_id = ?", (node, node)):
                edges[edge_id] = (source_id, relation, target_id, json.loads(data))
                other = target_id if source_id == node else source_id
                if other not in seen:
                    seen.add(other)
                    next_frontier.append(other)
        frontier = next_frontier
    entities = {node: entity_row(node) for node in seen}
    relationships = [(edge_id, entities[s][1], relation, entities[t][1], data) for edge_id, (s, relation, t, data) in sorted(edges.items())]
    return [entities[start]] + [entities[e] for e in sorted(seen - {start})], relationships


def find_entities(query, k=10):
    """Return (id, name, type, data, similarity) for the k entities closest in meaning to the query."""
    refresh_indexes()
    if entity_index.ntotal == 0:
        return []
    scores, ids = entity_index.search(embed(query, task=ENTITY_TASK)[None, :], min(k, entity_index.ntotal))
    return [(*entity_row(int(node)), float(score)) for score, node in zip(scores[0], ids[0])]


def forget_entity(name, type=None):
    """Delete an entity and every relationship it's part of. Returns how many relationships went with it."""
    entity = entity_id(name, type)
    removed = memory_cursor.execute("DELETE FROM relationships WHERE source_id = ? OR target_id = ?", (entity, entity)).rowcount
    memory_cursor.execute("DELETE FROM entities WHERE id = ?", (entity,))
    memory_database.commit()
    entity_index.remove_ids(np.array([entity], dtype=np.int64))
    return removed


def forget_relationship(relationship_id):
    """Delete one relationship by its id. The entities it joined stay."""
    if memory_cursor.execute("DELETE FROM relationships WHERE id = ?", (relationship_id,)).rowcount == 0:
        raise ValueError(f"No relationship {relationship_id}")
    memory_database.commit()


dedupe_memories()  # clean up duplicates saved before deduplication existed
prune_memories()  # short-term memories may have aged out since the last run

# Thread any documents cached before threads existed.
for rowid, title, blob in memory_cursor.execute("SELECT rowid, title, embedding FROM documents WHERE thread_status IS NULL ORDER BY rowid").fetchall():
    auto_thread(rowid, np.frombuffer(blob, dtype=np.float32), title)


def save_report(report):
    memory_cursor.execute("INSERT INTO reports (report) VALUES (?)", (report,))
    memory_database.commit()


def recent_reports(count=3):
    """Return (report, created_at) for the newest autonomous research reports, newest first."""
    return memory_cursor.execute("SELECT report, created_at FROM reports ORDER BY rowid DESC LIMIT ?", (count,)).fetchall()


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
