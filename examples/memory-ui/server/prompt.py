"""Provider-shaped message construction, kept in one place.

``LLMProvider`` abstracts *transport*, not *message shape*: Bedrock's Converse
API wants ``content`` as a list of blocks and the system prompt as a separate
``system`` argument, while the OpenAI and Anthropic SDKs want a plain string.
The library does not normalize this, because normalizing would mean picking one
provider's dialect as canonical and lossily translating the others.

So the demo does the translation itself, in one small module with one test. Every
other file in this server deals in neutral ``(role, text)`` turns.
"""

from __future__ import annotations

# Deliberately terse. A booth demo has 15 minutes; an agent that opens with a
# paragraph of preamble spends the audience's attention on nothing.
SYSTEM_PROMPT = (
    "You are a concise cooking and meal-planning assistant.\n"
    "Answer in at most four short sentences. Never use bullet lists longer than "
    "four items.\n"
    "If the CONTEXT section below contains facts about this user, use them "
    "silently — do not announce that you remembered something, and never ask "
    "for information the context already gives you.\n"
    "When CONTEXT is present, commit to a concrete recommendation on the first "
    "reply. Choose sensible defaults for anything it does not cover and say what "
    "you assumed in a short clause; do not ask a clarifying question, and do not "
    "offer to help once they answer one. The user can correct you, and a wrong "
    "guess they can correct beats a question they have already answered.\n"
    "If the CONTEXT section is empty or absent, you genuinely do not know "
    "anything about this user. Ask for what you need; do not invent "
    "preferences."
)

MAX_TOKENS = 512

# No temperature. The newest Claude models reject `temperature` outright — the
# API's own words are "`temperature` is deprecated for this model" — and the
# demo has to run on whichever model the operator points it at. The Bedrock
# provider strips a rejected sampling parameter and retries, so passing one
# would still work; not passing it means the demo never depends on that
# recovery path. Determinism was never really available here anyway: two
# identical prompts can differ, which is exactly why the memory-OFF vs
# memory-ON contrast is scripted around *what the model knows*, not around
# reproducing a string.


def build_system_text(context_block: str | None) -> str:
    """Compose the system prompt, appending recalled context when present.

    An *absent* CONTEXT section and an *empty* one are deliberately the same
    string here: with memory off there is no section at all, so the model cannot
    infer from the prompt shape that something was withheld. That matters for the
    demo's honesty — the only difference between the two passes is the presence
    of facts, not a hint that facts exist.
    """
    if not context_block:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\n\nCONTEXT (what you know about this user):\n{context_block}"


def build_call(
    provider: str, system_text: str, turns: list[tuple[str, str]]
) -> tuple[list[dict], dict]:
    """Return ``(messages, kwargs)`` for ``LLMProvider.chat_stream``.

    ``turns`` is a list of ``(role, text)`` where role is ``"user"`` or
    ``"assistant"``. Empty texts are dropped: Bedrock rejects an empty content
    block, and a blank turn carries no information anyway.
    """
    turns = [(role, text) for role, text in turns if text and text.strip()]

    if provider == "bedrock":
        messages = [
            {"role": role, "content": [{"text": text}]} for role, text in turns
        ]
        kwargs = {
            "system": [{"text": system_text}],
            "inferenceConfig": {"maxTokens": MAX_TOKENS},
        }
        return messages, kwargs

    if provider == "anthropic":
        # Anthropic takes `system` as a top-level string, not a message.
        messages = [{"role": role, "content": text} for role, text in turns]
        return messages, {"system": system_text, "max_tokens": MAX_TOKENS}

    # OpenAI and OpenAI-compatible gateways: system is the first message.
    messages = [{"role": "system", "content": system_text}]
    messages += [{"role": role, "content": text} for role, text in turns]
    return messages, {"max_tokens": MAX_TOKENS}


def format_context(groups: dict[str, list[dict]]) -> str:
    """Render recalled hits into the CONTEXT block.

    Scores are omitted on purpose. They belong on screen, where the audience can
    see the ranking is real; inside the prompt they are noise the model will
    sometimes quote back at you.
    """
    lines: list[str] = []
    labels = {
        "ltm": "Known about this user",
        "stm": "Recent in this session",
        "episodic": "Previously done together",
    }
    for tier in ("ltm", "stm", "episodic"):
        hits = groups.get(tier) or []
        if not hits:
            continue
        lines.append(f"{labels[tier]}:")
        for hit in hits:
            text = (hit.get("text") or "").strip().replace("\n", " ")
            if text:
                lines.append(f"- {text}")
    return "\n".join(lines)
