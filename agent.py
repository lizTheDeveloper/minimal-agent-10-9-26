# /// script
# dependencies = ["openai", "mlx-lm", "faiss-cpu", "numpy"]
# ///
"""The world's most basic agent: read input, call the model, run any tools, repeat."""

import os

from openai import OpenAI

from prompt import build_messages
from database import save_conversation_history
from tools import SCHEMAS, run_tool_call, search_memory, add_memory

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",  # OpenRouter speaks the OpenAI API
    api_key=os.environ["OPENROUTER_API_KEY"],  # get one at https://openrouter.ai/keys
)
MODEL = os.getenv("MODEL", "anthropic/claude-sonnet-5.5")  # any model slug from https://openrouter.ai/models


def respond(history, user_input):
    """Answer one user message, looping while the model asks for tools."""
    messages = build_messages(history, user_input)  # prompt.py decides what the model sees
    while True:
        reply = client.chat.completions.create(model=MODEL, messages=messages, tools=SCHEMAS).choices[0].message
        messages.append(reply)  # keep the model's turn, including any tool requests
        if not reply.tool_calls:  # no tools requested: this is the final answer
            return reply.content
        for call in reply.tool_calls:  # run each tool the model asked for
            messages.append(run_tool_call(call))  # tools.py does the work; result goes back to the model


def main():
    """Chat loop in the terminal."""
    history = []  # short-term memory: just the transcript
    while True:
        user_input = input("you> ")
        if user_input in ("", "quit", "exit"):
            ## create a conversation summary before exiting
            summary = respond(history, "Summarize the conversation so far, frame it as a memory as it will be stored to the memory index.")
            add_memory(summary)
            break
        # MEMORY HOOK 5: Proactive Recall - go search memory based on what the user just said
        memory_results = search_memory(user_input)
        if memory_results != "No memories yet.":
            print("agent> (recalled from memory)")
            print(memory_results)
        
        answer = respond(history, user_input)
        print("agent>", answer)
        history += [{"role": "user", "content": user_input}, {"role": "assistant", "content": answer}]
        save_conversation_history(history)  # overwrite the saved transcript with the latest one
        # MEMORY HOOK 4: write. Save this exchange to long-term memory here (embed + store)
        



if __name__ == "__main__":
    main()
