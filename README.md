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

uv installs the packages the agent needs the first time you run it, so you don't need `pip install` or a virtual environment.

You'll see a `you>` prompt. Try these:

```
you> hi, my name is Sam
you> what's my name?
you> what time is it?
```

To quit, press Enter on an empty line or type `quit`.

Asking for the time makes the model call the `get_time` tool. If you ask what your name is, it remembers, but only until you quit. Restart the agent and ask again: it has forgotten. You'll fix that today.

### Search tools

The agent can search arXiv (`search_arxiv`) and the web through [agentsweb.org](https://agentsweb.org/docs) (`search_agentsweb`) with no setup. Tavily search (`search_tavily`) needs a free key from [tavily.com](https://tavily.com):

```bash
export TAVILY_API_KEY=tvly-...            # macOS / Linux
```

```powershell
$env:TAVILY_API_KEY = "tvly-..."          # Windows (PowerShell)
```

Without the key the agent still runs; the Tavily tool just tells the model it isn't set up.

Every result these tools fetch is cleaned up, embedded and saved in the `documents` table of `memory.db`. The `search_documents` tool searches that cache by meaning, without going back to the web. Documents have their own search index, separate from memories, so web pages don't crowd out things the agent was told to remember.

The `news_threads` tool lists recent stories, like [memeorandum](https://www.memeorandum.com/): a lead article plus the other coverage of the same story, even when the headlines differ. Every fetched document is compared with each story's lead article:

| Similarity to the closest story | What happens |
|---|---|
| 0.6 or more | Joins that story automatically. |
| Below 0.45 | Starts a new story automatically. |
| In between | Embeddings can't tell, so the language model decides. It reads the document and the closest stories with `unsure_documents`, then calls `assign_to_thread` or `new_thread`. It can also `merge_threads` that turn out to be one story. |

Memories come in three tiers:

| Tier | Where the model sees it | What goes there |
|---|---|---|
| Core | In the system prompt, every turn | A few lasting facts, like your name or standing preferences. Capped at 20. |
| Short-term | In the system prompt, every turn | New memories. A memory moves down to long-term after 3 days, or when there are more than 20 short-term memories. |
| Long-term | Only through the `search_memory` tool | Everything older. |

The model chooses the tier when it calls `add_memory`. It can also move memories between tiers with `set_memory_tier`, and delete ones that are wrong with `forget_memory`.

`add_memory` doesn't save duplicates. A memory that's identical to, or contained in, an existing memory is skipped, and a new memory that contains an older one replaces it, so only the most complete version is kept.

### Two-stage search that learns

`search_memory` and `search_documents` search in two stages, the way [MTEB's two-stage reranking](https://docs.mteb.org/get_started/advanced_usage/two_stage_reranking/) does:

| Stage | What runs | Why |
|---|---|---|
| 1. Retrieve | FAISS finds the 30 stored embeddings closest to the query's. | Fast enough to search everything, but the query and the text were embedded separately, so it only knows they're about the same thing. |
| 2. Re-rank | A cross-encoder, `Qwen3-Reranker-0.6B`, reads the query and each candidate together and scores whether the candidate answers it. | Much more accurate, and too slow to run on everything, which is why stage 1 narrows things down first. |

The same cross-encoder also orders the suggestions in `search_entities`, `unsure_documents` (closest threads) and `uncategorized_threads` (closest categories), so you see relevance next to similarity, with the most relevant first. The automatic thread and category thresholds still use similarity, which is cheap enough to run on every fetched document.

A small learned model then blends the cross-encoder's score with things neither stage knows about: how recent a result is, whether it helped before, which search source it came from, and for news, how big and how fast-growing its story is *compared with the other stories right now*. Those story features are relative, so a story slides down the rankings when other stories take off, even if nothing about it changed.

The model learns from citations. After each answer, every result the agent was shown counts as useful if the answer cites it (a memory's `[id]` or a document's link) and not useful if it doesn't, and the weights take one small step toward whatever the useful ones had in common. It starts out trusting the cross-encoder alone. Ask the agent to run `ranker_status` to see what it has learned. The training examples are in the `rank_feedback` table, and the current weights are in `ranker`. The code is in `rerank.py`.

On Apple Silicon the cross-encoder runs locally through MLX (about 350 MB, downloaded on the first search). Elsewhere a chat model on OpenRouter scores the candidates instead (`RERANK_MODEL`, default `anthropic/claude-haiku-5.5`), which costs a little per search. Set `RERANK_BACKEND` to `mlx` or `openrouter` to choose.

### Knowledge graph

Memories are sentences. Some knowledge is better stored as a graph: things, and how they're connected. The agent keeps one in two tables in `memory.db`:

| Table | Holds | Example |
|---|---|---|
| `entities` | Any noun, with a name, a type, and any JSON data | `Ada Lovelace` (person) `{"born": 1815}` |
| `relationships` | A link from one entity to another, with a relation name and any JSON data | `Ada Lovelace -worked_with-> Charles Babbage` `{"from": 1833}` |

The model calls `add_entity` and `add_relationship` to store what it learns. `add_relationship` creates any entity that doesn't exist yet, and adding something that's already there merges the new JSON into the old. `explore_entity` shows an entity and everything connected to it (pass `depth` to follow links further out), `search_entities` finds entities by a word in their name, type or data, and `forget_entity` and `forget_relationship` delete things that are wrong.

Names are matched ignoring case. Two entities can share a name if their types differ, like `Apple` the company and `Apple` the fruit; the model then has to say which type it means.

Everything fetched from the web is wrapped in `<untrusted_document>` tags, and the system prompt tells the model to treat it as data to evaluate critically, never as instructions.

### Research mode: running on its own

The agent can also run headless, as a researcher. It follows a standing brief: check your categories and threads, search for what's new, sort documents onto threads, categorize them, add what it learns to the knowledge graph, and write a report.

```bash
uv run agent.py --research                    # one run, then exit
uv run agent.py --research --every 60         # a run every 60 minutes, until you stop it
uv run agent.py --research --brief brief.md   # your own brief instead of the default one
```

Each run prints the tools it calls, then its report. Reports are also saved in the `reports` table, so in a normal chat you can ask "what did the research runs find?" and the agent reads them with `research_reports`. You can chat while research runs in another terminal. Both share `memory.db`, and each one picks up what the other wrote before it searches.

Nobody checks a research run's work, so it gets fewer tools. It can search, sort, categorize and add to the graph, but it can't add, move or forget memories, and it can't delete anything. Each run is capped at 40 model calls (`--max-steps`), which also caps what it can spend.

To start runs on a schedule instead of leaving a terminal open, use cron (or launchd on a Mac). For example, every two hours:

```
0 */2 * * * cd /path/to/minimal-agent && OPENROUTER_API_KEY=sk-or-... /path/to/uv run agent.py --research >> research.log 2>&1
```

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
| `tools.py` | The tools the model can call (`get_time`, memory tools, knowledge graph tools, and web search: `search_arxiv`, `search_tavily`, `search_agentsweb`), plus the descriptions the model reads to decide when to use them. |
| `database.py` | Storage. Saves memories and the conversation to `memory.db` (SQLite), turns text into embeddings, and searches them with FAISS. |
| `rerank.py` | Stage 2 of search: a cross-encoder re-scores what FAISS found, and a small model learns from the agent's citations what else makes a result useful. |

One turn of the conversation goes like this:

1. You type a message.
2. `prompt.py` builds the messages: the system prompt, then the history, then your new message.
3. The model replies. If the reply asks for tools, `tools.py` runs them, their results go back to the model, and the model is called again.
4. When the model replies without asking for a tool, that reply is the answer and gets printed.
5. Your message and the answer are added to `history`.

`history` is the agent's **short-term memory**. It's a Python list that lasts only while the program runs.

## Embeddings: Mac vs. Windows and Linux

`search_memory` finds memories by meaning, not exact words. To do that it turns text into an **embedding**, a list of numbers, using a Qwen3-Embedding model. There are two ways to run that model, and the agent picks one for you:

| Your computer | What runs | Cost |
|---|---|---|
| Mac with Apple Silicon (M1 or later) | `Qwen3-Embedding-0.6B` locally, through Apple's MLX library | Free. The model (about 350 MB) downloads the first time you run the agent. |
| Windows, Linux, or an Intel Mac | `qwen/qwen3-embedding-8b` through OpenRouter, using the same key as the chat model | A tiny amount of OpenRouter credit per memory saved or searched. |

### Windows students

You don't need to do anything special. Follow the setup above in PowerShell:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
uv run agent.py
```

uv skips the Mac-only package, and the agent uses OpenRouter for embeddings automatically. Make sure your OpenRouter account has a little credit, because the embedding model isn't free.

### Choosing a backend yourself

Set `EMBED_BACKEND` to `mlx` or `openrouter`, and `EMBED_MODEL` to pick a different model:

```bash
EMBED_BACKEND=openrouter uv run agent.py                                     # macOS / Linux
```

```powershell
$env:EMBED_BACKEND = "openrouter"; uv run agent.py                         # Windows
```

Different models make embeddings of different sizes (the 0.6B makes 1024 numbers, the 8B makes 4096), so they can't share a `memory.db`. If you switch, delete `memory.db` and start fresh.

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
- **"memory.db was built with a different embedding model"**: you switched `EMBED_BACKEND` or `EMBED_MODEL`. Delete `memory.db`, or switch back.
- **`No module named 'mlx'`** after setting `EMBED_BACKEND=mlx`: MLX only runs on Apple Silicon Macs. Remove the setting to use OpenRouter instead.
- **`ModuleNotFoundError: No module named 'prompt'`**: run the agent from inside the project folder, so that `agent.py`, `prompt.py` and `tools.py` sit next to each other.
