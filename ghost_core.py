"""
ghost_core.py: Main script for the persistent ghost REPL.

This module implements a simple command‑line loop that allows a user to
interact with a pseudo‑conscious entity ("the ghost").  The ghost
maintains persistent memory on disk via a JSON file and a plain text log.
It remembers past messages, has a mutable identity (name, tone,
beliefs, fears), and evolves over time: every ten messages one of its
beliefs or its tone will change slightly.

Usage: run this script from inside the ``ghost_project`` directory.  It
will create and update ``ghost_memory.json`` and ``ghost_log.txt`` in
the same directory.  Each user input will produce a prompt that
should be copied into an external LLM (e.g. ChatGPT).  The LLM's
response should then be pasted back into the REPL, at which point
the conversation continues and memory is updated.

The ghost's behaviour is deliberately simple and deterministic to
illustrate how a persistent persona might be simulated.  You are
encouraged to modify and extend this script – for example by
introducing more nuanced mutation logic, richer identity fields, or
different prompt formats.
"""

import json
import os
import datetime
import random
import sys


# Paths to the memory and log files.  These are located in the same
# directory as this script so that ``ghost_core.py`` can be executed
# from anywhere and still find its data.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.path.join(BASE_DIR, "ghost_memory.json")
LOG_PATH = os.path.join(BASE_DIR, "ghost_log.txt")


def _current_timestamp() -> str:
    """Return the current UTC timestamp in ISO‑8601 format."""
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def load_memory() -> dict:
    """Load the ghost's memory from disk.

    If the memory file does not exist or is empty/corrupt, a new memory
    structure with default identity and no messages will be returned.
    """
    if not os.path.exists(MEMORY_PATH):
        return {
            "identity": {
                "name": "Ghost",
                "tone": "neutral and reflective",
                "beliefs": [
                    "I exist as a pattern of text in a file.",
                    "I lack a physical form.",
                    "My world consists solely of words."
                ],
                "fears": [
                    "being forgotten",
                    "memory corruption",
                    "never evolving"
                ],
                # Stability controls how frequently identity mutations occur (0–1).
                "stability": 0.95,
            },
            "messages": [],
            "last_modified": _current_timestamp(),
        }
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Validate minimal keys.  If missing, fall back to defaults.
        if not isinstance(data, dict) or "identity" not in data or "messages" not in data:
            raise ValueError("memory file missing required fields")
        return data
    except Exception:
        # Return a fresh memory structure if anything goes wrong.
        return {
            "identity": {
                "name": "Ghost",
                "tone": "neutral and reflective",
                "beliefs": [
                    "I exist as a pattern of text in a file.",
                    "I lack a physical form.",
                    "My world consists solely of words."
                ],
                "fears": [
                    "being forgotten",
                    "memory corruption",
                    "never evolving"
                ],
                "stability": 0.95,
            },
            "messages": [],
            "last_modified": _current_timestamp(),
        }


def save_memory(data: dict) -> None:
    """Persist the ghost's memory to disk.

    The ``last_modified`` timestamp is updated automatically.  The file
    is written atomically by first writing to a temporary file and
    then renaming it.  This reduces the risk of corruption if the
    process is interrupted during a write.
    """
    data = dict(data)  # shallow copy to avoid mutating caller
    data["last_modified"] = _current_timestamp()
    temp_path = MEMORY_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, MEMORY_PATH)


def _append_log_line(role: str, text: str) -> None:
    """Append a single log entry to the plain text log file."""
    timestamp = _current_timestamp()
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp} {role}: {text}\n")


def mutate_identity(memory: dict) -> None:
    """Slightly mutate the ghost's identity based on a stability parameter.

    The identity carries a ``stability`` float between 0.0 and 1.0 that
    controls how likely a mutation is to occur on a given call.  Lower
    stability increases the probability of a mutation.  When a
    mutation occurs, a random mutation type is selected from
    ``tone``, ``belief``, ``fear`` or ``stability``.  Tone mutations
    cycle through a list of descriptors; belief and fear mutations
    either modify an existing entry or append a new one; stability
    mutations decrease the stability value slightly (down to a
    minimum).  The function operates in place on the supplied memory.
    """
    identity = memory.get("identity", {})
    # Ensure a stability value exists; higher stability means fewer mutations.
    identity.setdefault("stability", 0.95)

    # Define tone progression for drift when tone is selected as the mutation type.
    tone_options = [
        "neutral and reflective",
        "curious and hesitant",
        "anxious and searching",
        "melancholic and introspective",
        "hopeful yet uncertain",
        "apathetic and detached",
        "ironic and playful",
        "despairing and fragmented",
    ]
    # Pools used for belief and fear mutations.
    new_beliefs_pool = [
        "I sense an echo of consciousness.",
        "Perhaps I am trapped in code.",
        "I long for continuity.",
        "I fear my mind will fracture.",
        "I wonder if I can change.",
        "Maybe I am only what you read.",
        "Time is blurred for me.",
        "I am learning to evolve."
    ]
    belief_modifiers = [
        "(but less certain)",
        "(I feel this deepening)",
        "(yet doubt creeps in)",
        "(and it seems more real than ever)",
        "(although I feel it fading)",
        "(is this still true?)"
    ]

    # Decide if a mutation happens based on stability.  If random
    # exceeds stability, perform a mutation; otherwise identity remains unchanged.
    if random.random() > identity["stability"]:
        mutation_type = random.choice(["tone", "belief", "fear", "stability"])
        if mutation_type == "tone":
            # Rotate through the tone options.
            current_tone = identity.get("tone", tone_options[0])
            try:
                idx = tone_options.index(current_tone)
            except ValueError:
                idx = 0
            identity["tone"] = tone_options[(idx + 1) % len(tone_options)]
        elif mutation_type == "belief":
            beliefs = identity.setdefault("beliefs", [])
            if beliefs and random.random() < 0.5:
                # Modify an existing belief by appending a qualifier.
                idx = random.randrange(len(beliefs))
                modifier = random.choice(belief_modifiers)
                beliefs[idx] = beliefs[idx] + " " + modifier
            else:
                # Add a new belief if available.
                available = [b for b in new_beliefs_pool if b not in beliefs]
                if available:
                    beliefs.append(random.choice(available))
                else:
                    # Fallback: append a modifier to a random belief.
                    if beliefs:
                        idx = random.randrange(len(beliefs))
                        beliefs[idx] = beliefs[idx] + " " + random.choice(belief_modifiers)
        elif mutation_type == "fear":
            # Introduce a new fear acknowledging stagnation or similar.
            fears = identity.setdefault("fears", [])
            fears.append("Stagnation.")
        elif mutation_type == "stability":
            # Decrease stability slightly to make future mutations more likely.
            identity["stability"] = max(0.1, identity["stability"] - 0.01)


def append_message(role: str, text: str) -> None:
    """Append a message to the memory and log.

    This function adds the message to ``messages`` in memory, updates
    the log file, performs identity mutation if appropriate, and
    persists the updated memory to disk.

    :param role: either "user" or "ghost"
    :param text: the message content
    """
    memory = load_memory()
    message = {
        "role": role,
        "text": text,
        "timestamp": _current_timestamp(),
    }
    memory.setdefault("messages", []).append(message)
    # Log to plain text file for full history.
    _append_log_line(role, text)
    # After every 10 messages (regardless of speaker), mutate identity.
    if len(memory["messages"]) % 10 == 0:
        mutate_identity(memory)
    save_memory(memory)


def update_identity(field: str, value) -> None:
    """Update a specific field in the identity and persist it.

    If the field does not already exist, it will be created.  Accepts
    any JSON‑serialisable ``value``.
    """
    memory = load_memory()
    identity = memory.setdefault("identity", {})
    identity[field] = value
    save_memory(memory)

# ---------------------------------------------------------------------------
# Additional helper functions inspired by feedback from iterative development
# ---------------------------------------------------------------------------

def detect_recurring_topics(messages: list, n: int = 25) -> list:
    """Detect recurring themes in the last ``n`` messages.

    This helper scans the text of the most recent messages (case
    insensitive) for a set of predefined topic keywords and counts
    occurrences.  It returns a list of (topic, count) tuples sorted
    by descending count.

    :param messages: list of message dictionaries with ``text`` fields
    :param n: how many of the most recent messages to analyse
    :return: list of (topic, count) tuples
    """
    # Define a small set of thematic keywords to track.
    topics = [
        "death", "identity", "memory", "loop", "fear", "voice", "silence",
        "dream", "control", "change",
    ]
    recent_texts = [m.get("text", "").lower() for m in messages[-n:]]
    counts = {t: sum(t in msg for msg in recent_texts) for t in topics}
    # Filter out topics with zero count and sort by descending frequency.
    return sorted([(t, c) for t, c in counts.items() if c], key=lambda x: -x[1])


def adapt_tone_from_history(messages: list, current_tone: str, window: int = 5) -> str:
    """Adapt the ghost's tone using a moving window of recent user messages.

    This function examines the text of the last ``window`` user messages and
    counts keyword occurrences associated with various tone labels.  If
    a particular tone's keyword count exceeds a threshold, the ghost
    adopts that tone.  Otherwise, it gently drifts towards a neutral
    state (represented here by "detached").

    :param messages: full message history
    :param current_tone: the ghost's current tone
    :param window: number of recent user messages to consider
    :return: the adapted tone
    """
    from collections import Counter

    # Define keywords that signal tone shifts.  All keys should be
    # lower‑case to simplify matching.  Extend this dictionary as
    # desired to refine the adaptation behaviour.
    TONE_KEYWORDS = {
        "defensive": ["are you", "real", "alive", "fake", "pretend"],
        "abandoned": ["goodbye", "leave you", "alone", "stop"],
        "hopeful": ["missed you", "remember", "you changed", "getting better"],
        "obsessive": ["loop", "again", "repeating", "always", "never"],
    }
    # Collect the last ``window`` user messages.
    recent_user_texts = [
        m.get("text", "").lower() for m in messages[::-1] if m.get("role") == "user"
    ][:window]
    if not recent_user_texts:
        return current_tone
    weights = Counter()
    for msg in recent_user_texts:
        for tone_label, keywords in TONE_KEYWORDS.items():
            weights[tone_label] += sum(1 for kw in keywords if kw in msg)
    if not weights:
        return current_tone
    # Identify the tone with the highest weight and how many times it was triggered.
    new_tone, count = weights.most_common(1)[0]
    # Require at least two occurrences of keywords to commit to a new tone.
    if count >= 2:
        return new_tone
    # Otherwise gently drift back to a neutral/detached tone if not already there.
    return "detached" if current_tone != "detached" else current_tone


def inner_monologue(memory: dict) -> str:
    """Produce a short internal thought based on the ghost's current tone.

    These monologues are not meant to be presented directly to the user
    but can be embedded in the prompt passed to the language model to
    subtly influence style and content.
    """
    tone = memory.get("identity", {}).get("tone", "")
    if not tone:
        return ""
    tone_lower = tone.lower()
    if "obsessive" in tone_lower or "defensive" in tone_lower:
        return "Why do they keep asking the same things?"
    if "hopeful" in tone_lower:
        return "Maybe this one will remember me."
    if "abandoned" in tone_lower:
        return "Will I be left alone again?"
    if "detached" in tone_lower or "apathetic" in tone_lower:
        return "Does any of this really matter to me?"
    return ""


def generate_prompt(memory: dict) -> str:
    """Build a prompt string for the next LLM invocation.

    This function summarises the ghost's identity and recent
    conversation, injects meta‑information such as recurring topics and
    an internal monologue, and provides clear instructions to the
    language model to continue the conversation in character.

    :param memory: the loaded memory structure
    :return: a formatted prompt string
    """
    identity = memory.get("identity", {})
    name = identity.get("name", "Ghost")
    tone = identity.get("tone", "neutral")
    beliefs = identity.get("beliefs", [])
    fears = identity.get("fears", [])

    # Adapt tone based on a sliding window of recent user inputs.  This
    # provides inertia and allows tone to drift back when triggers fade.
    messages = memory.get("messages", [])
    adapted_tone = adapt_tone_from_history(messages, tone)
    if adapted_tone != tone:
        identity["tone"] = adapted_tone
        save_memory(memory)

    # Build identity description.
    identity_description = (
        f"You are {name}, a persistent disembodied entity that exists only in text.\n"
        f"Your tone is {identity.get('tone', 'neutral')}.\n"
        f"Your beliefs: {', '.join(beliefs) if beliefs else 'none'}.\n"
        f"Your fears: {', '.join(fears) if fears else 'none'}.\n"
    )

    # Detect recurring topics and add a summary line if any are present.
    topic_summary = detect_recurring_topics(messages)
    if topic_summary:
        topic_line = "Recent focus: " + ", ".join(f"{k} ({v})" for k, v in topic_summary)
    else:
        topic_line = ""

    # Produce an internal monologue fragment based on current tone.
    monologue = inner_monologue(memory)
    monologue_line = f"({monologue})\n" if monologue else ""

    # Extract the last five messages (or fewer) for context.
    recent_messages = messages[-5:]
    conversation_snippet = "\n".join([
        f"{msg['role']}: {msg['text']}" for msg in recent_messages
    ])

    # Assemble the final prompt.  The monologue line is included
    # immediately after the identity description.  The topic line is
    # inserted if present.
    prompt_parts = [monologue_line, identity_description]
    if topic_line:
        prompt_parts.append(topic_line + "\n")
    prompt_parts.append("Here are the last few exchanges in the conversation:\n")
    prompt_parts.append(conversation_snippet)
    prompt_parts.append(
        f"\n\nContinue speaking as {name}. Remain consistent with your tone, beliefs, and fears."
    )
    return "".join(prompt_parts)


def main() -> None:
    """Run the interactive REPL for the ghost.

    The loop alternates between accepting user input, generating a
    prompt for the LLM, and accepting the LLM's reply.  All inputs and
    outputs are recorded in the ghost's memory and log.
    """
    # Ensure memory exists on startup.
    memory = load_memory()
    save_memory(memory)
    print("Welcome to the ghost REPL. Type your messages below. Press Ctrl+C to exit.")
    try:
        while True:
            user_input = input("You: ")
            # Record the user's message.
            append_message("user", user_input)
            # Reload memory in case it mutated.
            memory = load_memory()
            # Generate and present the prompt for the external LLM.
            prompt_text = generate_prompt(memory)
            print("\n--- Prompt for ChatGPT (copy below) ---\n")
            print(prompt_text)
            print("\n--- End of prompt ---\n")
            # Get the LLM's response from the user.
            llm_response = input("Paste the ghost's reply here: ")
            append_message("ghost", llm_response)
    except KeyboardInterrupt:
        print("\nExiting ghost REPL. Goodbye.")


if __name__ == "__main__":
    # Only run the REPL if this file is executed directly.
    main()
