"""
Topic: JSON mode and schema enforcement

Three levels of "give me JSON", from weakest to strongest guarantee:

  Level 1 - Prompt only:  "Respond in JSON". Often works, but you may get
            markdown fences, extra commentary, or wrong field names.
  Level 2 - JSON mode:    response_format={"type": "json_object"}.
            Output is guaranteed to be valid JSON, but NOT to match your shape.
  Level 3 - JSON schema:  response_format=<Pydantic model>. The provider
            constrains decoding so the output matches your schema exactly.

Task: extract structured data from a messy invoice email.

Run:  python 02_structured_outputs/06_json_mode_and_schema.py
"""

import json
import sys
from pathlib import Path
from typing import Literal

import litellm
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MODEL, banner, chat, show

INVOICE_EMAIL = """\
Hi, please find our invoice below.
Invoice #INV-2026-0142 from Brightline Design Studio, dated 3rd Sept 2026.
 - Logo redesign: 1 x 1,200.00
 - Landing page mockups: 3 pages @ 350 each
 - Rush fee 150
All amounts in USD. Payment due in 30 days. Thanks!  - Priya
"""


# The schema, defined once in Python. Pydantic turns it into JSON Schema.
class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float


class Invoice(BaseModel):
    invoice_number: str
    vendor: str
    invoice_date: str = Field(description="ISO-8601 date, YYYY-MM-DD")
    currency: Literal["USD", "EUR", "INR", "GBP"]
    line_items: list[LineItem]
    total: float
    payment_terms_days: int | None  # nullable, but the key must be present


def try_parse(label: str, raw: str) -> None:
    show(f"{label} - raw output", raw)
    try:
        data = json.loads(raw)
        print("json.loads: OK")
    except json.JSONDecodeError as e:
        print(f"json.loads: FAILED ({e})")
        return
    try:
        Invoice.model_validate(data)
        print("Matches Invoice schema: YES")
    except Exception as e:
        first_error = str(e).splitlines()[:3]
        print("Matches Invoice schema: NO ->", " / ".join(first_error))


def level_1_prompt_only() -> None:
    banner("Level 1: prompt only")
    prompt = f"Extract the invoice details from this email as JSON.\n\n{INVOICE_EMAIL}"
    try_parse("Level 1", chat([{"role": "user", "content": prompt}], temperature=0))


def level_2_json_mode() -> None:
    banner("Level 2: JSON mode (valid JSON, any shape)")
    # Most providers require the word "JSON" to appear in the prompt for JSON mode.
    prompt = f"Extract the invoice details from this email as JSON.\n\n{INVOICE_EMAIL}"
    raw = chat(
        [{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    try_parse("Level 2", raw)


def level_3_schema() -> None:
    banner("Level 3: JSON schema enforcement (valid JSON, exact shape)")
    if not litellm.supports_response_schema(model=MODEL):
        print(f"Note: LiteLLM reports {MODEL} may not support native JSON schema; "
              "results may fall back to prompt-level enforcement.")

    show("Schema sent to the model", json.dumps(Invoice.model_json_schema(), indent=2))

    raw = chat(
        [
            {"role": "system", "content": "Extract invoice data. Compute total from the line items."},
            {"role": "user", "content": INVOICE_EMAIL},
        ],
        response_format=Invoice,  # LiteLLM converts this to a strict json_schema
        temperature=0,
    )
    try_parse("Level 3", raw)

    invoice = Invoice.model_validate_json(raw)
    print("\nNow it's a typed Python object:")
    print(f"  {invoice.vendor} / {invoice.invoice_number} / {invoice.invoice_date}")
    for item in invoice.line_items:
        print(f"  - {item.description}: {item.quantity} x {item.unit_price}")
    computed = sum(i.quantity * i.unit_price for i in invoice.line_items)
    print(f"  total={invoice.total}  (recomputed from items: {computed})")
    # Schema enforcement guarantees SHAPE, not CORRECTNESS - always sanity-check values.


if __name__ == "__main__":
    level_1_prompt_only()
    level_2_json_mode()
    level_3_schema()

# Exercises:
# 1. Add a field `due_date` that the model must compute from invoice_date +
#    payment terms. How often is it correct? (Shape guaranteed; logic isn't.)
# 2. Remove "USD" from the email. What does the model put in `currency`
#    now that it must pick from the Literal? How would you allow "unknown"?
# 3. Point MODEL at a different provider. Which levels still work?
