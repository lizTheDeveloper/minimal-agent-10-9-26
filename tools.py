"""The tools the agent can call, and the code that runs them."""
import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from database import find_in_conversation_history, find_memories, store_memory


def get_time():
    """Return the current time. A toy tool so there's something to call."""
    from datetime import datetime  # imported here to keep the top of the file tiny
    return datetime.now().isoformat(timespec="minutes")


def search_memory(query, k=5):
    """MEMORY HOOK 3: recall as a tool. The model decides when to look something up."""
    results = find_memories(query, k)
    if not results:
        return "No memories yet."
    return "\n".join(f"{content} (created at {created_at}, similarity {score:.2f})" for content, created_at, score in results)

def add_memory(memory):
    """MEMORY HOOK 2: store as a tool. The model decides when to save something."""
    store_memory(memory)
    return "Memory added."

def search_conversation_history(query):
    """Keyword search over the saved conversation history."""
    matches = find_in_conversation_history(query)
    if not matches:
        return "No matching messages."
    return "\n".join(f"{message['role']}: {message['content']}" for message in matches)


def http_get(url, data=None, headers=None):
    """Fetch a URL (POST if data is given) and return the response body as text."""
    request = urllib.request.Request(url, data=data, headers={"User-Agent": "minimal-agent", **(headers or {})})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def search_arxiv(query, max_results=5):
    """Search arXiv papers. Free, no key. The API returns Atom XML."""
    search = " AND ".join(f"all:{word}" for word in query.split())  # every word must match
    params = urllib.parse.urlencode({"search_query": search, "max_results": max_results, "sortBy": "relevance"})
    feed = ET.fromstring(http_get(f"https://export.arxiv.org/api/query?{params}"))
    atom = {"a": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in feed.findall("a:entry", atom):
        title = " ".join(entry.find("a:title", atom).text.split())  # titles come with line breaks
        summary = " ".join(entry.find("a:summary", atom).text.split())[:500]
        papers.append(f"{title}\n{entry.find('a:id', atom).text}\n{summary}")
    return "\n\n".join(papers) or "No papers found."


def search_tavily(query, max_results=5):
    """Search the web with Tavily. Needs TAVILY_API_KEY (free tier at https://tavily.com)."""
    if "TAVILY_API_KEY" not in os.environ:
        return "Tavily search isn't set up: TAVILY_API_KEY is missing."
    body = json.dumps({"query": query, "max_results": max_results}).encode()
    headers = {"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}", "Content-Type": "application/json"}
    results = json.loads(http_get("https://api.tavily.com/search", data=body, headers=headers))["results"]
    return "\n\n".join(f"{r['title']}\n{r['url']}\n{r['content'][:500]}" for r in results) or "No results."


def search_agentsweb(query, count=5):
    """Search the web with agentsweb.org. Free, no key; snippets come back as markdown."""
    params = urllib.parse.urlencode({"q": query, "count": count})
    results = json.loads(http_get(f"https://agentsweb.org/web?{params}"))["results"]
    return "\n\n".join(f"{r['title']}\n{r['url']}\n{r['snippet'][:500]}" for r in results) or "No results."


TOOLS = {  # name -> python function
    "get_time": get_time,
    "search_memory": search_memory,
    "add_memory": add_memory,
    "search_conversation_history": search_conversation_history,
    "search_arxiv": search_arxiv,
    "search_tavily": search_tavily,
    "search_agentsweb": search_agentsweb,
}

SCHEMAS = [  # what the model is told about each tool (OpenAI-style function schemas)
    {"type": "function", "function": {
        "name": "get_time",
        "description": "Get the current date and time.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "search_memory",
        "description": "Search past notes and conversations.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "add_memory",
        "description": "Add a new memory to the long-term store.",
        "parameters": {"type": "object", "properties": {"memory": {"type": "string"}}, "required": ["memory"]},
    }},
    {"type": "function", "function": {
        "name": "search_conversation_history",
        "description": "Search the saved conversation history for messages containing a keyword or phrase.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "search_arxiv",
        "description": "Search arXiv for academic papers. Returns titles, links and abstracts.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "search_tavily",
        "description": "Search the web with Tavily. Returns titles, links and page summaries.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "search_agentsweb",
        "description": "Search the web with agentsweb.org. Returns titles, links and markdown snippets.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
]


def run_tool_call(call):
    """Run one tool call from the model and return the message that carries its result back."""
    args = json.loads(call.function.arguments or "{}")  # the model sends arguments as a JSON string
    result = TOOLS[call.function.name](**args)  # look up the function by name and call it
    return {"role": "tool", "tool_call_id": call.id, "content": str(result)}  # id ties result to the call
