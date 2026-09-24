"""
Topic: Chain-of-thought (CoT) for reasoning tasks

For multi-step problems, asking the model to reason step by step before the
final answer usually improves accuracy, because each generated token can
build on the previous steps. We ask for the final answer on a marked line
so our code can still extract it reliably.

Note: "reasoning" models (o-series, Claude with thinking, Gemini thinking)
do this internally. CoT prompting matters most for non-reasoning models,
and making the reasoning visible also helps with debugging and auditing.

Run:  python 01_prompt_engineering/04_chain_of_thought.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ask, banner, show

PROBLEM = """\
A SaaS plan costs $49 per user per month.
A team of 12 users pays annually and gets a 15% discount on the annual price.
Halfway through the year, 3 more users join. They are billed monthly, with no
discount, for the remaining 6 months.
Two of the original 12 users leave after 4 months, but annual seats are
non-refundable, so the team still pays for them.
What is the team's total spend for the year, in dollars?"""

# Compute the ground truth in code - never trust the model to grade itself.
annual = 12 * 49 * 12 * (1 - 0.15)   # 5997.60 (leavers still paid for)
joiners = 3 * 49 * 6                 # 882.00
EXPECTED = round(annual + joiners, 2)  # 6879.60

DIRECT_PROMPT = PROBLEM + "\n\nReply with only the final number. No explanation."

COT_PROMPT = PROBLEM + """

Work through this step by step:
1. List each cost component separately.
2. Calculate each component, showing the arithmetic.
3. Check whether any detail in the problem is a distractor.
4. Add them up.
On the last line, write: ANSWER: <number>"""


def extract_number(text: str) -> float | None:
    """Take the number after 'ANSWER:' if present, otherwise the last number."""
    match = re.search(r"ANSWER:\s*\$?(\d[\d,]*(?:\.\d+)?)", text)
    if not match:
        numbers = re.findall(r"\$?(\d[\d,]*(?:\.\d+)?)", text)
        if not numbers:
            return None
        return float(numbers[-1].replace(",", ""))
    return float(match.group(1).replace(",", ""))


def run(label: str, prompt: str, trials: int = 3) -> None:
    banner(label)
    hits = 0
    for i in range(1, trials + 1):
        # temperature > 0 so repeated trials show the variance
        output = ask(prompt, temperature=0.7)
        value = extract_number(output)
        ok = value is not None and abs(value - EXPECTED) < 0.01
        hits += ok
        if i == 1:
            show("Sample output (trial 1)", output)
        print(f"Trial {i}: extracted={value}  {'CORRECT' if ok else 'WRONG'}")
    print(f"Accuracy: {hits}/{trials}  (expected {EXPECTED})")


if __name__ == "__main__":
    run("Direct answer (no reasoning)", DIRECT_PROMPT)
    run("Chain-of-thought", COT_PROMPT)

# Exercises:
# 1. Run with a small/cheap model (e.g. MODEL=gpt-4o-mini or ollama/llama3.1)
#    and a reasoning model. Where does CoT make the biggest difference?
# 2. Replace the numbered steps with just "Think step by step." Compare.
# 3. CoT costs more output tokens (and latency). Print
#    response.usage.completion_tokens for both variants to measure the tradeoff.
