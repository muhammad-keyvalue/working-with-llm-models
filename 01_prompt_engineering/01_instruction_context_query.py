"""
Topic: Instruction -> Context -> Query structure

A well-structured prompt separates three things:
  1. INSTRUCTION - what the model should do and how (role, rules, output format)
  2. CONTEXT     - the data it should work from (documents, records, policies)
  3. QUERY       - the specific question for this request

We send the same question twice: once as an unstructured blob, once with
the three parts clearly delimited. Compare accuracy, grounding and format.

Run:  python 01_prompt_engineering/01_instruction_context_query.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ask, banner, show

REFUND_POLICY = """\
Acme Cloud Refund Policy (v3.2)
- Monthly plans: refundable within 7 days of the billing date.
- Annual plans: pro-rated refund within 30 days of purchase; no refund after 30 days.
- Add-ons (extra storage, premium support) are non-refundable.
- Refunds are issued to the original payment method within 5-10 business days.
- Accounts suspended for Terms of Service violations are not eligible for refunds.
"""

QUESTION = (
    "I bought the annual plan 20 days ago plus the premium support add-on. "
    "Can I get my money back, and how long will it take?"
)

# --- Version A: everything mashed together --------------------------------
unstructured_prompt = f"{REFUND_POLICY} {QUESTION} answer this for the customer"

# --- Version B: Instruction -> Context -> Query ---------------------------
structured_prompt = f"""\
## Instruction
You are a billing support assistant for Acme Cloud.
Answer the customer's question using ONLY the policy in the Context section.
If the policy does not cover something, say so; do not guess.
Respond in this format:
  Eligible: <Yes / No / Partial>
  Explanation: <2-3 sentences quoting the relevant policy points>
  Timeline: <when the money arrives, or "N/A">

## Context
<policy>
{REFUND_POLICY}
</policy>

## Query
<customer_question>
{QUESTION}
</customer_question>
"""

if __name__ == "__main__":
    banner("A) Unstructured prompt")
    show("Prompt", unstructured_prompt)
    show("Response", ask(unstructured_prompt, temperature=0))

    banner("B) Instruction -> Context -> Query")
    show("Prompt", structured_prompt)
    show("Response", ask(structured_prompt, temperature=0))

    print(
        "\nWhat to look for:\n"
        " - Does B correctly say 'Partial' (annual plan refundable, add-on is not)?\n"
        " - Is B's output format predictable enough to parse or show in a UI?\n"
        " - Did A invent anything that is not in the policy?"
    )

# Exercises:
# 1. Change the question to something the policy doesn't cover (e.g. "Can I
#    transfer my plan to a colleague?"). Does B admit it doesn't know?
# 2. Move the Query ABOVE the Context in version B. Does anything change?
#    (For long contexts, putting the question at the end usually works better.)
# 3. Replace the XML-style tags with ``` fences. Does the delimiter style matter?
