# /// script
# dependencies = ["flask"]
# ///
"""A live browser view of memory.db: the knowledge graph and the news threads. Run it next to the agent:

    uv run viewer.py        then open http://localhost:5050

It only reads the database, so it never gets in the agent's way, and the page refreshes itself whenever the agent writes."""

import json
import os
import sqlite3

from flask import Flask, jsonify, send_from_directory

DATABASE = os.getenv("MEMORY_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.db"))
app = Flask(__name__)


def query(sql, params=()):
    """Run one read-only query and return rows as dicts. Missing tables (an older database) just give no rows."""
    connection = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(sql, params)]
    except sqlite3.OperationalError as error:
        if "no such table" in str(error):
            return []
        raise
    finally:
        connection.close()


def parse(data):
    try:
        return json.loads(data or "{}")
    except json.JSONDecodeError:
        return {"raw": data}


@app.get("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "viewer.html")


@app.get("/api/version")
def version():
    """Changes whenever the agent commits, so the page knows to re-fetch."""
    stamps = [os.stat(path).st_mtime_ns for path in (DATABASE, DATABASE + "-wal") if os.path.exists(path)]
    return jsonify(version=max(stamps, default=0))


@app.get("/api/graph")
def graph():
    entities = query("SELECT id, name, type, data, created_at FROM entities ORDER BY id")
    relationships = query("SELECT id, source_id, relation, target_id, data, created_at FROM relationships ORDER BY id")
    for row in entities + relationships:
        row["data"] = parse(row["data"])
    return jsonify(entities=entities, relationships=relationships)


@app.get("/api/news")
def news():
    threads = query("SELECT id, title, lead_rowid, created_at FROM threads ORDER BY id")
    categories = query("SELECT thread_id, name, status FROM thread_categories JOIN categories ON categories.id = category_id")
    documents = query("SELECT rowid, url, title, substr(content, 1, 600) AS excerpt, source, query, fetched_at, "
                      "thread_id, thread_status FROM documents ORDER BY fetched_at DESC, rowid DESC")
    by_thread = {thread["id"]: {**thread, "categories": [], "documents": []} for thread in threads}
    for category in categories:
        if category["thread_id"] in by_thread:
            by_thread[category["thread_id"]]["categories"].append({"name": category["name"], "status": category["status"]})
    unthreaded = []
    for document in documents:
        thread = by_thread.get(document["thread_id"])
        if thread is None:
            unthreaded.append(document)
        elif document["rowid"] == thread["lead_rowid"]:
            thread["documents"].insert(0, document)  # lead story first
        else:
            thread["documents"].append(document)
    ordered = sorted(by_thread.values(), key=lambda thread: (len(thread["documents"]), thread["id"]), reverse=True)
    return jsonify(threads=ordered, unthreaded=unthreaded)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5050"))
    print(f"Viewing {DATABASE} at http://localhost:{port}")
    app.run(port=port, debug=False)
