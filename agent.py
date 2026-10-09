# /// script
# dependencies = ["openai"]
# ///
"""The world's most basic agent: read input, call the model, run any tools, repeat."""

import os

from openai import OpenAI

from prompt import build_messages
from tools import SCHEMAS, run_tool_call

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
            break
        answer = respond(history, user_input)
        print("agent>", answer)
        history += [{"role": "user", "content": user_input}, {"role": "assistant", "content": answer}]
        # MEMORY HOOK 4: write. Save this exchange to long-term memory here (embed + store)


if __name__ == "__main__":
    main()
