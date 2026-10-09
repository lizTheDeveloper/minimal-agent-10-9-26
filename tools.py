"""The tools the agent can call, and the code that runs them."""
import json

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


TOOLS = {"get_time": get_time, "search_memory": search_memory, "add_memory": add_memory, "search_conversation_history": search_conversation_history}  # name -> python function

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
]


def run_tool_call(call):
    """Run one tool call from the model and return the message that carries its result back."""
    args = json.loads(call.function.arguments or "{}")  # the model sends arguments as a JSON string
    result = TOOLS[call.function.name](**args)  # look up the function by name and call it
    return {"role": "tool", "tool_call_id": call.id, "content": str(result)}  # id ties result to the call
