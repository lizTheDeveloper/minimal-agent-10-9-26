# /// script
# dependencies = [
#   "openai",
#   "faiss-cpu",
#   "numpy",
#   "prompt_toolkit",
#   "transformers",  # Prompt Guard 2, a local prompt-injection classifier (see guard.py)
#   "torch",
#   "mlx-lm; sys_platform == 'darwin' and platform_machine == 'arm64'",  # local embeddings, Apple Silicon only
# ]
# ///
"""The world's most basic agent: read input, call the model, run any tools, repeat."""

import argparse
import os
import time
from datetime import datetime

from openai import OpenAI
from prompt_toolkit import PromptSession

from prompt import RESEARCH_BRIEF, build_messages
from database import save_conversation_history, save_report
from rerank import learn
from tools import AUTONOMOUS_SCHEMAS, AUTONOMOUS_TOOLS, SCHEMAS, run_tool_call, search_memory, add_memory

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",  # OpenRouter speaks the OpenAI API
    api_key=os.environ["OPENROUTER_API_KEY"],  # get one at https://openrouter.ai/keys
)
MODEL = os.getenv("MODEL", "anthropic/claude-sonnet-5.5")  # any model slug from https://openrouter.ai/models


def respond(history, user_input, autonomous=False, max_steps=None):
    """Answer one user message, looping while the model asks for tools. An autonomous run gets fewer tools, and
    max_steps caps how many times the model is called, so a run can't loop (and spend) forever."""
    messages = build_messages(history, user_input, autonomous)  # prompt.py decides what the model sees
    schemas, allowed = (AUTONOMOUS_SCHEMAS, AUTONOMOUS_TOOLS) if autonomous else (SCHEMAS, None)
    step = 0
    while True:
        step += 1
        if max_steps and step > max_steps:  # out of steps: one last call, with no tools, for the answer
            messages.append({"role": "user", "content": "You're out of steps. Stop researching and write your final report now."})
            return client.chat.completions.create(model=MODEL, messages=messages).choices[0].message.content
        reply = client.chat.completions.create(model=MODEL, messages=messages, tools=schemas).choices[0].message
        messages.append(reply)  # keep the model's turn, including any tool requests
        if not reply.tool_calls:  # no tools requested: this is the final answer
            return reply.content
        for call in reply.tool_calls:  # run each tool the model asked for
            if autonomous:
                print(f"  {call.function.name} {call.function.arguments}", flush=True)  # a log of what the run did
            messages.append(run_tool_call(call, allowed))  # tools.py does the work; result goes back to the model


def research(brief, every=None, max_steps=40):
    """Headless mode: follow the research brief with nobody watching, save the report, and repeat every few minutes."""
    while True:
        print(f"[{datetime.now():%Y-%m-%d %H:%M}] research run starting", flush=True)
        try:
            report = respond([], brief, autonomous=True, max_steps=max_steps)
            learn(report)  # its citations teach the ranker too
            save_report(report)  # the chat agent reads these with research_reports
            print(report, flush=True)
        except Exception as error:  # a flaky API shouldn't end a run that's meant to go all week; try again next time
            print(f"Research run failed: {error}", flush=True)
        if not every:
            return
        time.sleep(every * 60)


def main():
    """Chat loop in the terminal."""
    history = []  # short-term memory: just the transcript
    session = PromptSession()  # unlike input(), a multi-line paste arrives as one message
    while True:
        user_input = session.prompt("you> ")
        if user_input in ("quit", "exit"):
            ## create a conversation summary before exiting
            summary = respond(history, "Summarize the conversation so far, frame it as a memory as it will be stored to the memory index.")
            add_memory(summary)
            break
        # MEMORY HOOK 5: Proactive Recall - go search memory based on what the user just said
        memory_results = search_memory(user_input, record=False)  # only you see these, so they can't teach the ranker
        if memory_results != "No memories yet.":
            print("agent> (recalled from memory)")
            print(memory_results)
        
        answer = respond(history, user_input)
        learn(answer)  # search results the answer cited were useful; the rest weren't. See rerank.py
        print("agent>", answer)
        history += [{"role": "user", "content": user_input}, {"role": "assistant", "content": answer}]
        save_conversation_history(history)  # overwrite the saved transcript with the latest one
        # MEMORY HOOK 4: write. Save this exchange to long-term memory here (embed + store)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with the agent, or run it headless as a researcher.")
    parser.add_argument("--research", action="store_true", help="run headless: follow the research brief, save a report, exit")
    parser.add_argument("--every", type=float, metavar="MINUTES", help="with --research, keep running, once every MINUTES")
    parser.add_argument("--brief", metavar="FILE", help="with --research, a file with your own research brief")
    parser.add_argument("--max-steps", type=int, default=40, help="with --research, the most model calls per run (default 40)")
    args = parser.parse_args()
    if args.research:
        research(open(args.brief).read() if args.brief else RESEARCH_BRIEF, args.every, args.max_steps)
    else:
        main()
