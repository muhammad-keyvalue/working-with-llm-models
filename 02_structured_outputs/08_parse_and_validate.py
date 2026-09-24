"""
Topic: Parsing and validating model outputs in production code

Even with JSON mode or schemas, treat model output like any untrusted
external input: parse defensively, validate against business rules,
and have a plan for when it's wrong. The layers:

  1. Extract - pull JSON out of fences / surrounding prose
  2. Parse   - json.loads
  3. Validate- Pydantic: types, enums, ranges, cross-field business rules
  4. Repair  - on failure, send the error back to the model and retry (bounded)
  5. Fallback- if still failing, return a safe default and log it; never crash

Part 1 runs OFFLINE against canned "bad model outputs" - no API key needed.
Part 2 runs the full pipeline against a real model.

Run:  python 02_structured_outputs/08_parse_and_validate.py            (both parts)
      python 02_structured_outputs/08_parse_and_validate.py --offline  (part 1 only)
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import banner, chat

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("llm-output")


# ---------------------------------------------------------------------------
# The contract our application needs from the model
# ---------------------------------------------------------------------------
class TicketTriage(BaseModel):
    category: Literal["billing", "bug", "account", "feature_request"]
    priority: Literal["P1", "P2", "P3", "P4"]
    summary: str = Field(min_length=10, max_length=200)
    confidence: float = Field(ge=0.0, le=1.0)
    customer_email: str | None = None
    needs_human: bool

    @field_validator("category", mode="before")
    @classmethod
    def normalise_category(cls, value):
        # Tolerate harmless variation ("Billing", "feature request") instead of failing.
        if isinstance(value, str):
            return value.strip().lower().replace(" ", "_").replace("-", "_")
        return value

    @field_validator("customer_email")
    @classmethod
    def check_email(cls, value):
        if value is not None and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
            raise ValueError("not a valid email address")
        return value

    @model_validator(mode="after")
    def business_rules(self):
        # Cross-field rule the schema alone can't express.
        if self.priority == "P1" and not self.needs_human:
            raise ValueError("P1 tickets must set needs_human=true")
        return self


FALLBACK = TicketTriage(
    category="bug", priority="P3", summary="Automatic triage failed; needs manual review.",
    confidence=0.0, needs_human=True,
)


# ---------------------------------------------------------------------------
# Layers 1-3: extract, parse, validate
# ---------------------------------------------------------------------------
def extract_json(text: str) -> str:
    """Strip ```json fences and surrounding prose; return the outermost {...}."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in model output")
    return text[start : end + 1]


def parse_triage(raw: str) -> TicketTriage:
    """Raises ValueError / ValidationError with a readable message on failure."""
    data = json.loads(extract_json(raw))
    return TicketTriage.model_validate(data)


def describe_error(err: Exception) -> str:
    if isinstance(err, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'root'}: {e['msg']}" for e in err.errors())
    return str(err)


# ---------------------------------------------------------------------------
# Part 1: offline demo with realistic bad outputs
# ---------------------------------------------------------------------------
CANNED_OUTPUTS = {
    "clean JSON":
        '{"category": "billing", "priority": "P2", "summary": "Customer was charged twice in March.", '
        '"confidence": 0.92, "customer_email": "sam@example.com", "needs_human": false}',
    "wrapped in markdown fence + chatter":
        'Sure! Here is the triage:\n```json\n{"category": "Billing", "priority": "P2", '
        '"summary": "Customer was charged twice in March.", "confidence": 0.9, '
        '"customer_email": null, "needs_human": false}\n```\nLet me know if you need more!',
    "invalid JSON (trailing comma)":
        '{"category": "bug", "priority": "P1", "summary": "Files disappeared after sync.",}',
    "wrong enum + out-of-range number":
        '{"category": "outage", "priority": "urgent", "summary": "Site is down for everyone.", '
        '"confidence": 1.7, "customer_email": null, "needs_human": true}',
    "invalid email in a field":
        '{"category": "account", "priority": "P3", "summary": "Customer wants to change login email.", '
        '"confidence": 0.8, "customer_email": "not-an-email", "needs_human": false}',
    "valid types, but breaks a business rule":
        '{"category": "bug", "priority": "P1", "summary": "All project files deleted after sync.", '
        '"confidence": 0.8, "customer_email": "ops@bigcorp.io", "needs_human": false}',
    "no JSON at all":
        "I'm sorry, I can't classify this ticket without more information.",
}


def offline_demo() -> None:
    banner("Part 1 (offline): parsing real-world model output")
    for label, raw in CANNED_OUTPUTS.items():
        try:
            result = parse_triage(raw)
            print(f"\n[OK]   {label}\n       -> {result.model_dump()}")
        except (ValueError, ValidationError) as err:
            # json.JSONDecodeError is a subclass of ValueError
            print(f"\n[FAIL] {label}\n       -> {describe_error(err)}")


# ---------------------------------------------------------------------------
# Layers 4-5: repair loop + fallback, with a real model
# ---------------------------------------------------------------------------
SYSTEM = f"""\
You triage customer support tickets. Reply with ONLY a JSON object matching this schema:
{json.dumps(TicketTriage.model_json_schema())}
Rules: P1 means data loss or full outage and always requires needs_human=true.
"""


def triage(ticket: str, max_attempts: int = 3) -> TicketTriage:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ticket}]
    for attempt in range(1, max_attempts + 1):
        raw = chat(messages, temperature=0)
        try:
            result = parse_triage(raw)
            log.info("attempt %d: valid output", attempt)
            return result
        except (ValueError, ValidationError) as err:
            error = describe_error(err)
            log.warning("attempt %d: invalid output (%s)", attempt, error)
            # Repair: show the model its own output and exactly what was wrong.
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"Your reply failed validation: {error}. "
                           "Reply again with ONLY the corrected JSON object.",
            })
    log.error("all %d attempts failed; using fallback", max_attempts)
    return FALLBACK


def live_demo() -> None:
    banner("Part 2 (live): triage with validate -> repair -> fallback")
    tickets = [
        "Hi, this is sam@example.com. I was billed twice for my March subscription.",
        "URGENT!!! after your update last night ALL our project files are gone. "
        "Contact me at ops@bigcorp.io",
        "would be cool if the dashboard had keyboard shortcuts",
    ]
    for ticket in tickets:
        print(f"\nTicket: {ticket}")
        result = triage(ticket)
        print(f"Result: {result.model_dump()}")
        # Downstream code can now trust types and rules:
        if result.needs_human:
            print("-> routed to on-call human queue")


if __name__ == "__main__":
    offline_demo()
    if "--offline" not in sys.argv:
        live_demo()

# Exercises:
# 1. Make the SYSTEM prompt stricter or pass response_format=TicketTriage (see
#    06_json_mode_and_schema.py). Which failure types disappear, and which
#    (business rules!) can still happen?
# 2. Add a rule: confidence < 0.5 must set needs_human=true. Test it offline
#    by adding a canned output that violates it.
# 3. Log the number of repair attempts per request. In production this is a
#    key quality metric - a rising repair rate often signals prompt or model drift.
