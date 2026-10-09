"""Builds the messages we send to the model. The easiest place to put memory: right into the prompt."""

SYSTEM = "You are a helpful assistant. Use tools when they help."  # the agent's standing instructions


def build_messages(history, user_input):
    """Turn the conversation so far plus the new user message into the list the model sees."""
    # MEMORY HOOK 1: recall here, then put it in the system prompt (e.g. SYSTEM + "\nRelevant notes:\n" + recalled)
    system = {"role": "system", "content": SYSTEM}  # always first
    user = {"role": "user", "content": user_input}  # the new message
    # MEMORY HOOK 2: or inject recalled docs as an extra message just before `user`
    return [system] + history + [user]  # history is everything said so far
