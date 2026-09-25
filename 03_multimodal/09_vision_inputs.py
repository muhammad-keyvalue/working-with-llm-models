"""
Topic: Working with vision-capable models (image + text inputs)

A vision model takes images as part of the user message. In the OpenAI
format that LiteLLM uses, the message content becomes a LIST of parts:

    {"role": "user", "content": [
        {"type": "text", "text": "What is the total on this receipt?"},
        {"type": "image_url", "image_url": {"url": "https://... or data:image/png;base64,..."}},
    ]}

Local files are sent as base64 data URLs (see image_part() in common.py).
Images are billed as tokens too. The "detail" setting trades accuracy for cost.

The sample images are drawn in code with Pillow, so the correct answers are
known and every demo is scored rather than eyeballed.

Run:  python 03_multimodal/09_vision_inputs.py
      python 03_multimodal/09_vision_inputs.py path/to/image.png   (ask about your own image)
"""

import sys
from pathlib import Path

import litellm
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import VISION_MODEL, banner, image_part, show

SAMPLES = Path(__file__).parent / "samples"

SIGNUPS = {"Jan": 420, "Feb": 380, "Mar": 510, "Apr": 470, "May": 640, "Jun": 590}
SIGNUPS_THIS_WEEK = {**SIGNUPS, "Mar": 290, "Jun": 720}  # two bars changed

RECEIPT_ITEMS = [  # (item, quantity, unit price)
    ("Flat white", 2, 4.20),
    ("Almond croissant", 1, 3.85),
    ("Masala chai", 1, 3.40),
    ("Avocado toast", 1, 9.75),
    ("Sparkling water", 3, 2.10),
    ("Banana bread", 2, 3.15),
]
TAX_RATE = 0.08
SUBTOTAL = round(sum(q * p for _, q, p in RECEIPT_ITEMS), 2)
TOTAL = round(SUBTOTAL * (1 + TAX_RATE), 2)


# ---------------------------------------------------------------------------
# Sample images, drawn in code
# ---------------------------------------------------------------------------
def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def draw_chart(values: dict[str, int], title: str, path: Path) -> Path:
    width, height, left, top, bottom, max_value = 720, 460, 70, 70, 400, 800
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((left, 20), title, fill="black", font=font(24))
    for tick in range(0, max_value + 1, 200):
        y = bottom - (bottom - top) * tick / max_value
        draw.line([(left, y), (width - 30, y)], fill="#dddddd")
        draw.text((left - 45, y - 8), str(tick), fill="#555555", font=font(14))
    slot = (width - 30 - left) / len(values)
    for i, (label, value) in enumerate(values.items()):
        x0, x1 = left + i * slot + slot * 0.2, left + (i + 1) * slot - slot * 0.2
        y = bottom - (bottom - top) * value / max_value
        draw.rectangle([x0, y, x1, bottom], fill="#4C72B0")
        draw.text((x0 + 8, y - 20), str(value), fill="black", font=font(14))
        draw.text((x0 + 12, bottom + 8), label, fill="black", font=font(16))
    img.save(path)
    return path


def draw_receipt(path: Path) -> Path:
    # A big canvas with small text, like a phone photo of a receipt.
    img = Image.new("RGB", (1000, 1500), "#b9b4a8")
    draw = ImageDraw.Draw(img)
    x0, x1, y = 330, 670, 140
    draw.rectangle([x0 - 30, 100, x1 + 30, 760], fill="white")

    def line(left: str, right: str = "", size: int = 15) -> None:
        nonlocal y
        draw.text((x0, y), left, fill="black", font=font(size))
        draw.text((x1, y), right, fill="black", font=font(size), anchor="ra")
        y += size + 11

    line("CORNER CAFE", size=22)
    line("12 Market St  -  2026-09-14 08:42")
    line("-" * 60)
    for name, qty, price in RECEIPT_ITEMS:
        line(f"{qty} x {name}", f"{qty * price:.2f}")
    line("-" * 60)
    line("Subtotal", f"{SUBTOTAL:.2f}")
    line(f"Tax {TAX_RATE:.0%}", f"{TOTAL - SUBTOTAL:.2f}")
    line("TOTAL", f"{TOTAL:.2f}", size=20)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Output schemas (response_format, as in program 6)
# ---------------------------------------------------------------------------
class Bar(BaseModel):
    label: str
    value: int


class ChartData(BaseModel):
    title: str
    bars: list[Bar]


class ReceiptLine(BaseModel):
    item: str
    quantity: int
    amount: float


class Receipt(BaseModel):
    lines: list[ReceiptLine]
    subtotal: float
    total: float


class Change(BaseModel):
    label: str
    before: int
    after: int


class ChartDiff(BaseModel):
    changes: list[Change]


def ask_images(question: str, *images: Path, detail: str = "auto", **kwargs):
    """Send one text part + N image parts. Returns (text, prompt_tokens)."""
    content = [{"type": "text", "text": question}] + [image_part(p, detail) for p in images]
    response = litellm.completion(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        **kwargs,
    )
    return response.choices[0].message.content or "", response.usage.prompt_tokens


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------
def demo_vague_vs_precise(chart: Path) -> None:
    banner("1. Vague question vs precise, structured question")
    text, _ = ask_images("What does this image show?", chart)
    show("Vague: 'What does this image show?'", text)

    raw, _ = ask_images(
        "Read the chart title and the exact value printed above every bar.",
        chart,
        response_format=ChartData,
    )
    data = ChartData.model_validate_json(raw)
    read = {bar.label: bar.value for bar in data.bars}
    print("\n--- Precise + schema ---")
    for label, expected in SIGNUPS.items():
        got = read.get(label)
        print(f"  {label}: read={got}  expected={expected}  {'OK' if got == expected else 'WRONG'}")
    # A vague prompt gives a description. A precise prompt gives data your code can use.


def demo_detail_levels(receipt: Path) -> None:
    banner("2. detail='low' vs detail='high' on small text")
    tokens_used = {}
    for detail in ("low", "high"):
        raw, tokens = ask_images(
            "Transcribe every line item, the subtotal and the total from this receipt.",
            receipt,
            detail=detail,
            response_format=Receipt,
        )
        tokens_used[detail] = tokens
        receipt_data = Receipt.model_validate_json(raw)
        matched = sum(
            any(name.lower() in ln.item.lower() and abs(ln.amount - q * p) < 0.01 for ln in receipt_data.lines)
            for name, q, p in RECEIPT_ITEMS
        )
        print(f"\ndetail={detail:<4}  prompt_tokens={tokens}")
        print(f"  line items correct: {matched}/{len(RECEIPT_ITEMS)}")
        print(f"  total read: {receipt_data.total}  expected: {TOTAL}  "
              f"{'OK' if abs(receipt_data.total - TOTAL) < 0.01 else 'WRONG'}")
    if tokens_used["low"] == tokens_used["high"]:
        print(f"\nSame token count for both: {VISION_MODEL} (or the proxy in front of it) "
              "ignores `detail`. Check the cost of images on your own stack; don't assume it.")
    # 'low' shrinks the image to about 512px: cheap and fast, but small text blurs.
    # Use 'high' for documents, receipts and screenshots. 'low' is fine for "is there a cat?".


def demo_multi_image(last_week: Path, this_week: Path) -> None:
    banner("3. Several images in one request: what changed?")
    raw, _ = ask_images(
        "Image 1 is last week's report, image 2 is this week's. "
        "List every bar whose value changed, with the before and after values.",
        last_week,
        this_week,
        response_format=ChartDiff,
    )
    found = {(c.label, c.before, c.after) for c in ChartDiff.model_validate_json(raw).changes}
    expected = {(k, SIGNUPS[k], SIGNUPS_THIS_WEEK[k]) for k in SIGNUPS if SIGNUPS[k] != SIGNUPS_THIS_WEEK[k]}
    print(f"Model found: {sorted(found)}")
    print(f"Expected:    {sorted(expected)}")
    print("Result:", "CORRECT" if found == expected else "MISMATCH")
    # Images are referenced by position, so say which is which in the text part.


def demo_hallucination(chart: Path) -> None:
    banner("4. Asking about something that is NOT in the image")
    question = "How many signups were there in August?"
    text, _ = ask_images(question + " Answer with a number.", chart)
    show("Naive (forces a number)", text)

    text, _ = ask_images(
        question + "\nAnswer only from what is visible in the image. "
        "If the image does not show it, reply exactly: NOT_VISIBLE",
        chart,
    )
    show("With an explicit way out", text)
    # Vision models hallucinate like text models do: they extrapolate trends and
    # fill in unreadable text. Give them a sanctioned "I can't see it" answer.


def describe_own_image(path: str) -> None:
    banner(f"Your image: {path}")
    text, tokens = ask_images(
        "Describe this image. Then list any text you can read in it, exactly as written.",
        Path(path),
        detail="high",
    )
    show(f"Answer ({tokens} prompt tokens)", text)


if __name__ == "__main__":
    if not litellm.supports_vision(model=VISION_MODEL):
        print(f"Note: LiteLLM doesn't list {VISION_MODEL} as vision-capable "
              "(normal for proxy aliases). If requests fail, set VISION_MODEL in .env.")

    if len(sys.argv) > 1:
        describe_own_image(sys.argv[1])
        sys.exit()

    SAMPLES.mkdir(exist_ok=True)
    chart = draw_chart(SIGNUPS, "Monthly signups (last week)", SAMPLES / "chart_last_week.png")
    chart_new = draw_chart(SIGNUPS_THIS_WEEK, "Monthly signups (this week)", SAMPLES / "chart_this_week.png")
    receipt = draw_receipt(SAMPLES / "receipt.png")
    print(f"Sample images written to {SAMPLES}/ (open them to see what the model sees)")

    demo_vague_vs_precise(chart)
    demo_detail_levels(receipt)
    demo_multi_image(chart, chart_new)
    demo_hallucination(chart)

# Exercises:
# 1. Remove the value labels from the bars in draw_chart(), so the model has
#    to estimate from the gridlines. How far off is it? Would you trust a
#    vision model to read numbers off a chart without labels?
# 2. Shrink the receipt font to 11 and re-run demo 2. At what size does
#    detail='high' start failing too?
# 3. Swap the order of the two images in demo 3 without changing the text.
#    Does the model notice, or does it report the changes backwards?
