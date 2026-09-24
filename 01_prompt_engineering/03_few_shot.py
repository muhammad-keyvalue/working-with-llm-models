"""
Topic: Few-shot prompting

Zero-shot: describe the task and hope the model infers your exact format and
labelling conventions. Few-shot: show a handful of worked examples first.

Task: triage support tickets into "<CATEGORY> | <PRIORITY>" using OUR
team's conventions (which the model can't know without examples):
  - Anything mentioning data loss is always P1, even if the tone is calm.
  - Feature requests are always P4.

We check each output against the expected label and exact format.

Run:  python 01_prompt_engineering/03_few_shot.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import banner, chat

INSTRUCTION = (
    "Classify the support ticket. Categories: BILLING, BUG, ACCOUNT, FEATURE_REQUEST. "
    "Priorities: P1 (critical) to P4 (low). Reply with the label only."
)

# Few-shot examples: (ticket, label). Pick examples that teach the edge cases.
EXAMPLES = [
    ("I was charged twice for March.", "BILLING | P2"),
    ("Small thing, no rush: after the last sync a few of our project files are just gone.", "BUG | P1"),
    ("Would love a dark mode in the dashboard!", "FEATURE_REQUEST | P4"),
    ("I can't log in, password reset email never arrives.", "ACCOUNT | P2"),
    ("The export button has a typo in its tooltip.", "BUG | P4"),
]

# Test tickets with the label our team would expect.
TEST_CASES = [
    ("Hey, not urgent, but our uploaded invoices from last week seem to have vanished.", "BUG | P1"),
    ("Please add SSO support with Okta.", "FEATURE_REQUEST | P4"),
    ("Why did my bill go up by $40 this month?", "BILLING | P2"),
    ("I need to change the email address on my account.", "ACCOUNT | P3"),
    ("Dashboard graphs render slightly misaligned on Safari.", "BUG | P4"),
]

LABEL_FORMAT = re.compile(r"^(BILLING|BUG|ACCOUNT|FEATURE_REQUEST) \| P[1-4]$")


def zero_shot(ticket: str) -> str:
    return chat(
        [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": ticket},
        ],
        temperature=0,
    )


def few_shot(ticket: str) -> str:
    # Examples are sent as prior user/assistant turns, which models follow
    # very reliably. (Putting them inside one prompt as text also works.)
    messages = [{"role": "system", "content": INSTRUCTION}]
    for example_ticket, label in EXAMPLES:
        messages.append({"role": "user", "content": example_ticket})
        messages.append({"role": "assistant", "content": label})
    messages.append({"role": "user", "content": ticket})
    return chat(messages, temperature=0)


def evaluate(name: str, classify) -> None:
    banner(name)
    correct = well_formed = 0
    for ticket, expected in TEST_CASES:
        output = classify(ticket).strip()
        ok_format = bool(LABEL_FORMAT.match(output))
        ok_label = output == expected
        well_formed += ok_format
        correct += ok_label
        mark = "PASS" if ok_label else "FAIL"
        print(f"[{mark}] {ticket[:55]:<55} -> {output!r}  (expected {expected})")
    n = len(TEST_CASES)
    print(f"\nFormat valid: {well_formed}/{n}   Exact match: {correct}/{n}")


if __name__ == "__main__":
    evaluate("Zero-shot", zero_shot)
    evaluate("Few-shot (5 examples)", few_shot)

# Exercises:
# 1. Remove the "files are just gone" example. Does the data-loss -> P1 rule
#    still get applied to the first test case?
# 2. Make all examples BUG tickets. Watch the model's bias shift toward BUG.
#    (Lesson: examples should be diverse and cover the labels evenly.)
# 3. Add a ticket that fits two categories. How would you teach the tie-break
#    rule - with a sentence in the instruction, or with an example?
