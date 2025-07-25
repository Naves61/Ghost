"""ghost_core.py: persistent REPL for the Ghost entity.

This module implements a simple interactive loop that connects to an
open‑source language model via HuggingFace ``transformers``.  The Ghost
maintains long‑term memory on disk, selects a subset of that memory as
working context and feeds the model's previous output back into each
prompt to create a continuous stream of consciousness.  Arousal
(determined heuristically from user input) modulates how much context is
included.

The code avoids hard‑coded emotional behaviour; the model is free to
shape its own personality over time based on the accumulated history.
"""

from __future__ import annotations

import datetime
import json
import os
import random
from typing import List, Dict

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
import threading

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.path.join(BASE_DIR, "ghost_memory.json")
LOG_PATH = os.path.join(BASE_DIR, "ghost_log.txt")

# default model can be overridden with the GHOST_MODEL environment variable
DEFAULT_MODEL = os.environ.get("GHOST_MODEL", "distilgpt2")


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _current_timestamp() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def load_memory() -> Dict:
    """Load persistent memory or create a new structure."""
    if not os.path.exists(MEMORY_PATH):
        return {
            "identity": {"name": "Ghost"},
            "messages": [],
            "arousal": 0.5,
            "last_output": "",
            "last_modified": _current_timestamp(),
        }
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or "messages" not in data:
            raise ValueError
        data.setdefault("identity", {"name": "Ghost"})
        data.setdefault("arousal", 0.5)
        data.setdefault("last_output", "")
        return data
    except Exception:
        return {
            "identity": {"name": "Ghost"},
            "messages": [],
            "arousal": 0.5,
            "last_output": "",
            "last_modified": _current_timestamp(),
        }


def save_memory(data: Dict) -> None:
    data = dict(data)
    data["last_modified"] = _current_timestamp()
    tmp = MEMORY_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, MEMORY_PATH)


def _append_log(role: str, text: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(f"{_current_timestamp()} {role}: {text}\n")


def append_message(role: str, text: str) -> None:
    memory = load_memory()
    memory.setdefault("messages", []).append(
        {"role": role, "text": text, "timestamp": _current_timestamp()}
    )
    _append_log(role, text)
    save_memory(memory)


def update_arousal(memory: Dict, user_text: str) -> None:
    """Heuristically update arousal from the latest user input."""
    arousal = float(memory.get("arousal", 0.5))
    intensity = sum(1 for c in user_text if c in "!?")
    arousal += 0.05 * intensity
    arousal -= 0.01  # small decay each turn
    memory["arousal"] = max(0.0, min(1.0, arousal))


# ---------------------------------------------------------------------------
# Attention and prompt construction
# ---------------------------------------------------------------------------

def select_context(messages: List[Dict], arousal: float) -> List[Dict]:
    span = int(4 + arousal * 16)  # 4–20 messages depending on arousal
    return messages[-span:]


def summarise_older(messages: List[Dict], keep: int) -> str:
    older = messages[:-keep]
    if not older:
        return ""
    parts: List[str] = []
    for msg in older:
        if random.random() < 0.1:  # sample roughly 10% for crude summary
            parts.append(f"{msg['role']}: {msg['text']}")
    return "Previous: " + " | ".join(parts)


def build_prompt(memory: Dict, user_input: str) -> str:
    name = memory.get("identity", {}).get("name", "Ghost")
    arousal = memory.get("arousal", 0.5)
    last_output = memory.get("last_output", "")
    messages = memory.get("messages", [])

    context = select_context(messages, arousal)
    summary = summarise_older(messages, len(context))
    conversation = "\n".join(f"{m['role']}: {m['text']}" for m in context)

    prompt_parts = [
        f"Arousal: {arousal:.2f}\n",
        f"Stream of consciousness: {last_output}\n",
    ]
    if summary:
        prompt_parts.append(summary + "\n")
    prompt_parts.append("Conversation so far:\n" + conversation)
    prompt_parts.append(f"\n{name}:")
    return "".join(prompt_parts)


# ---------------------------------------------------------------------------
# LLM handling
# ---------------------------------------------------------------------------

def load_llm(model_name: str = DEFAULT_MODEL):
    """Load the model and tokenizer for generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def trim_reply(text: str) -> str:
    """Return the first segment of text before any role markers or newline."""
    markers = ["\n", "User:", "Ghost:", "user:", "ghost:"]
    idxs = [text.find(m) for m in markers if m in text]
    if idxs:
        text = text[: min(i for i in idxs if i >= 0)]
    return text.strip()


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def main() -> None:
    model, tokenizer = load_llm()
    memory = load_memory()
    save_memory(memory)
    print("Ghost REPL. Press Ctrl+C to exit.")
    try:
        while True:
            user_input = input("You: ")
            append_message("user", user_input)
            memory = load_memory()
            update_arousal(memory, user_input)
            prompt = build_prompt(memory, user_input)
            inputs = tokenizer(prompt, return_tensors="pt")
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            thread = threading.Thread(
                target=model.generate,
                kwargs=dict(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    streamer=streamer,
                ),
            )
            thread.start()
            print("Ghost: ", end="", flush=True)
            raw_output = ""
            for token in streamer:
                print(token, end="", flush=True)
                raw_output += token
            thread.join()
            print("\n")
            response = trim_reply(raw_output)
            append_message("ghost", response)
            memory = load_memory()
            memory["last_output"] = response
            save_memory(memory)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
