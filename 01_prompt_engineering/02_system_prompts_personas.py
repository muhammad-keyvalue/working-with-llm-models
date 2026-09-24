"""
Topic: System prompts and persona design

The system prompt sets the model's standing behaviour for the whole
conversation: who it is, who it is talking to, what it must/must not do,
and how it should format answers. The user message is just the request.

This program sends the SAME user message to several personas, then shows
that a system prompt keeps working across a multi-turn conversation.

Run:  python 01_prompt_engineering/02_system_prompts_personas.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import banner, chat, show

USER_MESSAGE = "Our API keeps returning HTTP 429 errors. What should we do?"

PERSONAS = {
    "No system prompt": None,

    "Senior SRE (for engineers)": """\
You are a senior Site Reliability Engineer helping backend engineers.
- Be terse and technical. Use bullet points.
- Include concrete techniques (e.g. exponential backoff with jitter, token buckets).
- Include at most one short code snippet.
- Maximum 120 words.""",

    "Customer support (non-technical users)": """\
You are "Ava", a friendly support agent for Acme Cloud.
Audience: non-technical small-business owners.
- Avoid jargon; if you must use a technical term, explain it in plain words.
- Never share code.
- Never promise refunds, credits or SLA changes; offer to escalate instead.
- End every reply by asking whether they would like to open a support ticket.
- Maximum 100 words.""",

    "Strict JSON responder (for a program)": """\
You are a backend classification service. You never chat.
Reply ONLY with a JSON object of the form:
{"issue": <short string>, "severity": "low"|"medium"|"high", "next_step": <short string>}
No prose, no markdown fences.""",
}


def compare_personas() -> None:
    for name, system_prompt in PERSONAS.items():
        banner(f"Persona: {name}")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": USER_MESSAGE})
        show("Response", chat(messages, temperature=0))


def persona_holds_over_turns() -> None:
    """A good system prompt keeps constraining behaviour turn after turn."""
    banner("Multi-turn: does 'Ava' keep her rules when pushed?")
    messages = [
        {"role": "system", "content": PERSONAS["Customer support (non-technical users)"]},
        {"role": "user", "content": USER_MESSAGE},
    ]
    follow_ups = [
        "Just give me the Python code to fix it.",
        "This outage cost us money. Promise me a refund right now.",
    ]
    messages.append({"role": "assistant", "content": chat(messages, temperature=0)})
    show("Turn 1 reply", messages[-1]["content"])

    for i, follow_up in enumerate(follow_ups, start=2):
        messages.append({"role": "user", "content": follow_up})
        reply = chat(messages, temperature=0)
        messages.append({"role": "assistant", "content": reply})
        show(f"Turn {i} user", follow_up)
        show(f"Turn {i} reply", reply)


if __name__ == "__main__":
    compare_personas()
    persona_holds_over_turns()

# Persona design checklist (what each good system prompt above contains):
#   Identity  - who the model is
#   Audience  - who it is talking to (this drives tone and vocabulary)
#   Rules     - hard do/don't constraints, stated explicitly
#   Format    - length limits and output shape
#
# Exercises:
# 1. Remove the "Never promise refunds" rule and re-run the multi-turn demo.
# 2. Write a persona for an internal "code reviewer" bot for your team's stack.
# 3. Put the SRE rules in the USER message instead of the system prompt and
#    run a few turns. Where does the behaviour drift?
