# Minimal Agent

This is about the simplest agent there is, in three small files. It reads what you type, sends it to a model, runs any tools the model asks for, and repeats until the model gives an answer.

Today you'll give it **memory**. The code has four places marked `MEMORY HOOK`, and those are where your work goes.

## Setup

You need two things:

1. **uv**, the Python tool we use to run the script. If `uv --version` doesn't work, install it:
   ```bash
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. **An OpenRouter API key.** Sign in at [openrouter.ai](https://openrouter.ai), then go to [openrouter.ai/keys](https://openrouter.ai/keys) and create a key. It starts with `sk-or-`.

Get the code:

```bash
git clone https://github.com/lizTheDeveloper/minimal-agent-10-9-26.git
cd minimal-agent-10-9-26
```

## Running it

Put your key in an environment variable, then run the agent:

```bash
# macOS / Linux
export OPENROUTER_API_KEY=sk-or-...
uv run agent.py
```

```powershell
# Windows (PowerShell)
$env:OPENROUTER_API_KEY = "sk-or-..."
uv run agent.py
```

uv installs the `openai` package the first time you run it, so you don't need `pip install` or a virtual environment.

You'll see a `you>` prompt. Try these:

```
you> hi, my name is Sam
you> what's my name?
you> what time is it?
```

To quit, press Enter on an empty line or type `quit`.

Asking for the time makes the model call the `get_time` tool. If you ask what your name is, it remembers, but only until you quit. Restart the agent and ask again: it has forgotten. You'll fix that today.

### Using a different model

The default model is `anthropic/claude-sonnet-5.5`. To try another one, pick any model ID from [openrouter.ai/models](https://openrouter.ai/models) and set `MODEL`:

```bash
MODEL=openai/gpt-4o-mini uv run agent.py
```

The model has to support tool calling. If you get errors about tools, try a different model.

## How it works

| File | What it does |
|---|---|
| `agent.py` | The main loop. It reads your input, calls the model, runs any tools the model asks for, and prints the answer. |
| `prompt.py` | Builds the list of messages the model sees: the system prompt, the conversation so far, and your new message. |
| `tools.py` | The tools the model can call (`get_time` and `search_memory`), plus the descriptions the model reads to decide when to use them. |

One turn of the conversation goes like this:

1. You type a message.
2. `prompt.py` builds the messages: the system prompt, then the history, then your new message.
3. The model replies. If the reply asks for tools, `tools.py` runs them, their results go back to the model, and the model is called again.
4. When the model replies without asking for a tool, that reply is the answer and gets printed.
5. Your message and the answer are added to `history`.

`history` is the agent's **short-term memory**. It's a Python list that lasts only while the program runs.

## Your task: add long-term memory

Search the code for `MEMORY HOOK`. Each hook is a different place memory can go:

| Hook | File | Idea |
|---|---|---|
| **1: recall into the system prompt** | `prompt.py` | Before each turn, look up relevant notes and add them to the system prompt. |
| **2: recall as a message** | `prompt.py` | Same as hook 1, but add the notes as a separate message just before the user's message. |
| **3: recall as a tool** | `tools.py` | Let the model decide when to search, by implementing `search_memory`. |
| **4: write** | `agent.py` | After each exchange, save it somewhere that lasts, such as a file or a vector store. |

Recall (hooks 1, 2 and 3) only works once there's something saved, so do hook 4 first.

### A suggested path

1. **Write to a file.** At hook 4, append each exchange to a JSON or text file.
2. **Recall everything.** At hook 1, read the whole file and paste it into the system prompt. Restart the agent and ask "what's my name?" It should remember now.
3. **Recall only what's relevant.** Pasting everything stops working once the file gets big. Change recall to search the file, first with plain keyword matching and then with embeddings and similarity search. That second step is RAG.
4. **Make it a tool.** Implement `search_memory` (hook 3) so the model searches only when it decides to. How is that different from always putting memory in the prompt?

### Questions to think about

- What should be saved: every message, or a summary, or only facts like "the user's name is Sam"?
- When does recall make the answers worse instead of better?
- What happens when the memory file says something that's no longer true?

## Troubleshooting

- **`KeyError: 'OPENROUTER_API_KEY'`**: the key isn't set in this terminal. Run the `export` (or `$env:`) line again in the same window you run the agent from.
- **`401` or "authentication" errors**: the key is wrong or has been revoked. Make a new one at [openrouter.ai/keys](https://openrouter.ai/keys).
- **`402` or "insufficient credits"**: your OpenRouter account needs credits, or you can switch to a free model (its ID ends in `:free`).
- **Model not found**: check the model ID on [openrouter.ai/models](https://openrouter.ai/models) and set it with `MODEL=...`.
- **`ModuleNotFoundError: No module named 'prompt'`**: run the agent from inside the project folder, so that `agent.py`, `prompt.py` and `tools.py` sit next to each other.
