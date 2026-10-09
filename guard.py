"""Prompt Guard 2: a small local classifier that scores how much text looks like a prompt injection or jailbreak."""
import os

# The 22M model (283 MB) is English-only; the 86M one (1.1 GB) is multilingual. Both need Meta's license accepted on Hugging Face.
GUARD_MODEL = os.getenv("GUARD_MODEL", "meta-llama/Llama-Prompt-Guard-2-22M")
GUARD_THRESHOLD = 0.5  # at or above this score, treat the text as an injection attempt
guard = None  # (tokenizer, model) once loaded, or False if Prompt Guard isn't available


def load_guard():
    """Load the classifier on first use. If it can't load (no Hugging Face access, no torch), the agent runs without it."""
    global guard
    if guard is None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            guard = (AutoTokenizer.from_pretrained(GUARD_MODEL), AutoModelForSequenceClassification.from_pretrained(GUARD_MODEL).eval())
        except Exception as error:
            print(f"(Prompt Guard is off: {str(error).splitlines()[0]})")
            guard = False
    return guard


def injection_score(text):
    """Return the probability (0 to 1) that text contains a prompt injection, or None if Prompt Guard isn't available.
    The model reads 512 tokens at a time, so long text is checked in overlapping windows and the highest score wins."""
    if not load_guard() or not text.strip():
        return None if not guard else 0.0
    import torch
    tokenizer, model = guard
    windows = tokenizer(text, truncation=True, max_length=512, stride=64, return_overflowing_tokens=True,
                        padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids=windows["input_ids"], attention_mask=windows["attention_mask"]).logits
    return float(torch.softmax(logits, dim=-1)[:, 1].max())  # label 1 is "malicious"


def is_injection(text):
    score = injection_score(text)
    return score is not None and score >= GUARD_THRESHOLD
