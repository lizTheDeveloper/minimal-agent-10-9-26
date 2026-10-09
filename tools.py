"""The tools the agent can call, and the code that runs them."""

import json


def get_time():
    """Return the current time. A toy tool so there's something to call."""
    from datetime import datetime  # imported here to keep the top of the file tiny
    return datetime.now().isoformat(timespec="minutes")


def search_memory(query):
    """MEMORY HOOK 3: recall as a tool. The model decides when to look something up."""
    return "No memories yet."  # replace with your vector store / keyword search


TOOLS = {"get_time": get_time, "search_memory": search_memory}  # name -> python function

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
]


def run_tool_call(call):
    """Run one tool call from the model and return the message that carries its result back."""
    args = json.loads(call.function.arguments or "{}")  # the model sends arguments as a JSON string
    result = TOOLS[call.function.name](**args)  # look up the function by name and call it
    return {"role": "tool", "tool_call_id": call.id, "content": str(result)}  # id ties result to the call
