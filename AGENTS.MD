**AGENTS.MD**

---

# Ghost Agent

The *Ghost* agent is a long‑running, text‑based entity designed to simulate a sense of continuity and internal life when interacting with an open‑source language model. It operates as a REPL, persisting its state to disk and using the model as a decoder to generate free‑form responses.

## Key Components

### Long‑Term Memory

All exchanges between the user and Ghost are appended to a persistent JSON file (`ghost_memory.json`). Each entry records the role (`user` or `ghost`), the text, and a timestamp. This log forms the agent’s long‑term memory and is used to reconstruct context across sessions.

To prevent unbounded growth, older entries are occasionally summarised. A configurable summariser (e.g. a sequence‑to‑sequence model from Hugging Face) compresses a block of past messages into a short synopsis. If summarisation is disabled or fails, the oldest messages are discarded once a threshold is reached.

### Short‑Term (Operative) Memory

For each prompt, only a subset of recent messages is included as working context. The span of this window is modulated by an arousal scalar (see below). At low arousal the agent focuses on the last few turns; at high arousal it draws from a larger slice of memory.

### Attention

The agent uses a simple retrieval heuristic to select which long‑term memories are relevant to the current user input. It scores past messages by lexical overlap with the current query and includes the top results along with recent messages and summaries. This mechanism allows Ghost to recall pertinent details without exceeding the model’s context limit.

### Arousal

Ghost maintains a floating‑point arousal value between 0 and 1. It affects both the size of the short‑term memory and the verbosity of the model’s response. Arousal is updated heuristically based on the intensity of the user’s input (e.g. the presence of exclamation marks) and decays slightly with each turn.

### Stream of Consciousness

The model’s previous output is stored as `last_output` and fed back into the next prompt. This loop encourages the model to reflect on its own words and maintain continuity. The agent does not hard‑code any personality; emergent behaviour arises from the model and the memory loop.

## Prompt Construction

When the user sends a message, Ghost builds a prompt containing:

1. The current arousal value and the previous output (stream of consciousness).
2. A summary of older exchanges, if available.
3. Selected short‑term and relevant long‑term messages, formatted as `<role>: <text>` lines.
4. A marker indicating that it is Ghost’s turn to speak (e.g. ending with `Ghost:`).

The prompt is passed to the loaded causal language model via Hugging Face’s `generate` method. The resulting text is trimmed at the first occurrence of a role marker to avoid the model speaking for the user. The trimmed response is streamed back to the console and appended to memory as the ghost’s reply.

## Model Integration

Ghost uses an open‑source causal model loaded through `AutoModelForCausalLM` and `AutoTokenizer`. The model name is configurable via an environment variable or command‑line flag. For interactive streaming, the implementation employs Hugging Face’s `TextIteratorStreamer` to print tokens in real time. The `pad_token_id` is set to the end‑of‑sequence token to suppress pad‑token warnings.

## Interaction Flow

1. **User Input:** The REPL reads a line from the user.
2. **Memory Update:** The input is appended to the memory log and arousal is adjusted.
3. **Prompt Assembly:** The agent constructs the prompt using the mechanisms above.
4. **Model Generation:** The model generates a response, streamed to the console. Generation stops when a new role marker is detected or the ghost asks a question.
5. **State Persistence:** The ghost’s reply and updated arousal are saved back to `ghost_memory.json`.
6. **Next Turn:** If the ghost’s output ends with a question mark, the REPL immediately prompts the user; otherwise, the ghost may continue to speak until it asks for input.

This architecture allows Ghost to behave like an entity with memory and self‑reflection, while remaining fully deterministic and auditable.
