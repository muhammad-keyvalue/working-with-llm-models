"""
Topic: Document understanding - PDFs, tables, unstructured content (and OCR)

"PDF" hides two very different things:
  - Digital PDF: has a text layer. Extract text directly: fast, free, exact
    characters, but table layout is flattened.
  - Scanned PDF: each page is just an image. No text to extract, so you need
    OCR: classic OCR (Tesseract, AWS Textract, Azure Document Intelligence) or
    a vision model reading the page image.

So OCR is not a separate topic. It is one step in a document pipeline:

  PDF -> has text layer? --yes--> extract text ------------------+
                         --no---> page image -> OCR / vision ----+-> LLM -> schema -> validate

The last two steps are the ones from Part 2: extract into a Pydantic schema,
then check business rules (the line items must add up to the total).

The sample invoice is generated in code twice, as a digital PDF and as a
"scanned" image-only PDF, so every extracted field can be checked.

Run:  python 03_multimodal/11_document_understanding.py
      python 03_multimodal/11_document_understanding.py path/to/invoice.pdf   (your own PDF)
"""

import base64
import random
import sys
from pathlib import Path

import litellm
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pydantic import BaseModel
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MODEL, VISION_MODEL, banner, chat, image_part, show

SAMPLES = Path(__file__).parent / "samples"

# Ground truth for the generated invoice
VENDOR = "Northwind Cloud Services"
INVOICE_NUMBER = "NW-88213"
INVOICE_DATE = "2026-09-02"
ITEMS = [  # (description, quantity, unit price)
    ("Compute instances (vCPU-hours)", 1250, 0.048),
    ("Block storage (GB-month)", 800, 0.10),
    ("Managed PostgreSQL", 2, 145.00),
    ("Data egress (GB)", 340, 0.09),
    ("Premium support", 1, 250.00),
]
TAX_RATE = 0.18
SUBTOTAL = round(sum(q * p for _, q, p in ITEMS), 2)
TAX = round(SUBTOTAL * TAX_RATE, 2)
TOTAL = round(SUBTOTAL + TAX, 2)
TERMS = (
    "Payment is due within 45 days of the invoice date. Balances unpaid after "
    "that incur interest of 1.5% per month. Please quote the invoice number "
    "on your transfer. Questions: billing@northwind.example"
)
PAYMENT_DAYS, LATE_INTEREST = 45, 1.5


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    amount: float


class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    invoice_date: str  # YYYY-MM-DD
    line_items: list[LineItem]
    subtotal: float
    tax: float
    total: float
    payment_terms_days: int | None  # from the terms paragraph (unstructured text)
    late_interest_percent_per_month: float | None


SYSTEM = (
    "Extract the invoice into the schema. Copy numbers exactly as printed. "
    "Dates as YYYY-MM-DD. Payment terms come from the terms paragraph."
)


# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------
def make_digital_pdf(path: Path) -> Path:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, VENDOR, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 7, f"Invoice {INVOICE_NUMBER}    Date: {INVOICE_DATE}    Currency: USD",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    with pdf.table(col_widths=(90, 30, 30, 30), text_align=("LEFT", "RIGHT", "RIGHT", "RIGHT")) as table:
        table.row(["Description", "Qty", "Unit price", "Amount"])
        for desc, qty, price in ITEMS:
            table.row([desc, f"{qty:g}", f"{price:.3f}", f"{qty * price:.2f}"])
    pdf.ln(3)
    for label, value in (("Subtotal", SUBTOTAL), (f"Tax ({TAX_RATE:.0%})", TAX), ("Total due", TOTAL)):
        pdf.cell(150, 7, label, align="R")
        pdf.cell(30, 7, f"{value:.2f}", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.multi_cell(0, 6, TERMS)
    pdf.output(str(path))
    return path


def make_scanned_pdf(path: Path) -> Path:
    """Render the invoice as a slightly rotated, noisy image and wrap it in a PDF."""
    font = lambda size: ImageFont.load_default(size=size)  # noqa: E731
    img = Image.new("L", (1240, 1754), 250)  # A4 at 150 dpi, grayscale
    draw = ImageDraw.Draw(img)
    y = 120
    draw.text((110, y), VENDOR, fill=20, font=font(40)); y += 70
    draw.text((110, y), f"Invoice {INVOICE_NUMBER}    Date: {INVOICE_DATE}    Currency: USD",
              fill=30, font=font(22)); y += 70
    cols = (110, 760, 930, 1130)
    for row in [("Description", "Qty", "Unit price", "Amount")] + [
        (d, f"{q:g}", f"{p:.3f}", f"{q * p:.2f}") for d, q, p in ITEMS
    ]:
        draw.text((cols[0], y), row[0], fill=30, font=font(22))
        for x, cell in zip(cols[1:], row[1:]):
            draw.text((x, y), cell, fill=30, font=font(22), anchor="ra")
        y += 44
        draw.line([(110, y - 8), (1130, y - 8)], fill=170)
    y += 20
    for label, value in (("Subtotal", SUBTOTAL), (f"Tax ({TAX_RATE:.0%})", TAX), ("Total due", TOTAL)):
        draw.text((930, y), label, fill=30, font=font(22), anchor="ra")
        draw.text((1130, y), f"{value:.2f}", fill=30, font=font(22), anchor="ra")
        y += 40
    y += 40
    words, line = TERMS.split(), ""
    for word in words:  # naive word wrap
        if len(line) + len(word) > 80:
            draw.text((110, y), line, fill=40, font=font(20)); y += 32; line = ""
        line += word + " "
    draw.text((110, y), line, fill=40, font=font(20))

    # Make it look scanned: slight rotation, blur, speckle noise.
    img = img.rotate(1.2, expand=False, fillcolor=235).filter(ImageFilter.GaussianBlur(0.8))
    rng = random.Random(7)
    pixels = img.load()
    for _ in range(25000):
        x, y = rng.randrange(img.width), rng.randrange(img.height)
        pixels[x, y] = rng.choice((90, 140, 200))
    scan = path.with_suffix(".jpg")
    img.save(scan, quality=60)

    pdf = FPDF()
    pdf.add_page()
    pdf.image(str(scan), x=0, y=0, w=210)  # the page is only an image: no text layer
    pdf.output(str(path))
    return path


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
def extract_text_layer(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def page_images(path: Path) -> list[Path]:
    """Pull the embedded page images out of a scanned PDF."""
    out = []
    for n, page in enumerate(PdfReader(path).pages, start=1):
        for img in page.images:
            target = SAMPLES / f"{path.stem}_p{n}_{img.name}"
            target.write_bytes(img.data)
            out.append(target)
    return out


def extract_from_text(text: str) -> Invoice:
    raw = chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        response_format=Invoice,
        temperature=0,
    )
    return Invoice.model_validate_json(raw)


def extract_from_images(images: list[Path]) -> Invoice:
    # This call is the OCR step: the vision model reads the pixels.
    content = [{"type": "text", "text": "Extract this invoice."}] + [image_part(p, "high") for p in images]
    response = litellm.completion(
        model=VISION_MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
        response_format=Invoice,
        temperature=0,
    )
    return Invoice.model_validate_json(response.choices[0].message.content)


def extract_from_pdf_file(path: Path) -> Invoice:
    # Some models accept the PDF itself (they get the text and page images).
    encoded = base64.b64encode(path.read_bytes()).decode()
    file_part = {"type": "file", "file": {"filename": path.name,
                                          "file_data": f"data:application/pdf;base64,{encoded}"}}
    response = litellm.completion(
        model=VISION_MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": [{"type": "text", "text": "Extract this invoice."}, file_part]}],
        response_format=Invoice,
        temperature=0,
    )
    return Invoice.model_validate_json(response.choices[0].message.content)


def process(path: Path) -> tuple[str, Invoice]:
    """Route by document type, then extract."""
    text = extract_text_layer(path)
    if len(text.strip()) > 50:
        return "text layer", extract_from_text(text)
    images = page_images(path)
    if not images:
        raise ValueError(f"{path} has no text layer and no images")
    return "vision OCR", extract_from_images(images)


def business_rule_errors(inv: Invoice) -> list[str]:
    """Checks that don't need ground truth, so they also work in production."""
    errors = []
    for item in inv.line_items:
        if abs(item.quantity * item.unit_price - item.amount) > 0.01:
            errors.append(f"{item.description}: {item.quantity} x {item.unit_price} != {item.amount}")
    if abs(sum(i.amount for i in inv.line_items) - inv.subtotal) > 0.01:
        errors.append(f"line items sum to {sum(i.amount for i in inv.line_items):.2f}, subtotal says {inv.subtotal}")
    if abs(inv.subtotal + inv.tax - inv.total) > 0.01:
        errors.append(f"subtotal + tax = {inv.subtotal + inv.tax:.2f}, total says {inv.total}")
    return errors


def ground_truth_score(inv: Invoice) -> str:
    checks = {
        "vendor": inv.vendor == VENDOR,
        "invoice_number": inv.invoice_number == INVOICE_NUMBER,
        "date": inv.invoice_date == INVOICE_DATE,
        "line_items": len(inv.line_items) == len(ITEMS)
        and all(abs(a.amount - q * p) < 0.01 for a, (_, q, p) in zip(inv.line_items, ITEMS)),
        "total": abs(inv.total - TOTAL) < 0.01,
        "payment_days": inv.payment_terms_days == PAYMENT_DAYS,
        "late_interest": inv.late_interest_percent_per_month == LATE_INTEREST,
    }
    wrong = [name for name, ok in checks.items() if not ok]
    return f"{len(checks) - len(wrong)}/{len(checks)} fields correct" + (f", wrong: {wrong}" if wrong else "")


def report(label: str, inv: Invoice) -> None:
    print(f"\n{label}")
    print(f"  {inv.vendor} / {inv.invoice_number} / {inv.invoice_date} / total {inv.total}")
    print(f"  terms: {inv.payment_terms_days} days, {inv.late_interest_percent_per_month}%/month late")
    errors = business_rule_errors(inv)
    print("  business rules:", "PASS" if not errors else "FAIL")
    for e in errors:
        print(f"    - {e}")
    print(f"  vs ground truth: {ground_truth_score(inv)}")


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------
def demo_look_inside(digital: Path, scanned: Path) -> None:
    banner("1. What's actually inside each PDF?")
    text = extract_text_layer(digital)
    show(f"{digital.name}: text layer ({len(text)} chars)", text)
    print("\nNote how the table became plain lines: column boundaries are gone.")
    scanned_text = extract_text_layer(scanned)
    print(f"\n{scanned.name}: text layer has {len(scanned_text.strip())} chars -> needs OCR")


def demo_naive_vs_schema(digital: Path) -> None:
    banner("2. Naive 'extract the table' vs schema + validation")
    text = extract_text_layer(digital)
    show("Naive", chat([{"role": "user", "content": f"Extract the table from this invoice:\n\n{text}"}]))
    print("\n(Readable, but your code still has to parse it.)")
    report("Schema extraction from the text layer", extract_from_text(text))


def demo_routes(digital: Path, scanned: Path) -> None:
    banner("3. The routed pipeline on both documents")
    for path in (digital, scanned):
        route, inv = process(path)
        report(f"{path.name} -> route: {route}", inv)
    # The scanned route usually costs more and makes more mistakes. The business
    # rules catch many misreads without knowing the right answer: a misread
    # digit usually breaks qty x price = amount, or the sum.


def demo_native_pdf(scanned: Path) -> None:
    banner("4. Sending the PDF file itself (if the model supports it)")
    try:
        report(f"{scanned.name} as a file part", extract_from_pdf_file(scanned))
    except Exception as e:
        print(f"{VISION_MODEL} rejected the PDF input: {type(e).__name__}: {str(e)[:200]}")
        print("That's why the pipeline above renders pages to images: it works with any vision model.")


if __name__ == "__main__":
    print(f"MODEL={MODEL}  VISION_MODEL={VISION_MODEL}")
    SAMPLES.mkdir(exist_ok=True)

    if len(sys.argv) > 1:
        own = Path(sys.argv[1])
        route, inv = process(own)
        print(f"\nRoute: {route}")
        print(inv.model_dump_json(indent=2))
        print("Business rule errors:", business_rule_errors(inv) or "none")
        sys.exit()

    digital = make_digital_pdf(SAMPLES / "invoice_digital.pdf")
    scanned = make_scanned_pdf(SAMPLES / "invoice_scanned.pdf")
    print(f"Sample PDFs written to {SAMPLES}/ (open them to compare)")

    demo_look_inside(digital, scanned)
    demo_naive_vs_schema(digital)
    demo_routes(digital, scanned)
    demo_native_pdf(scanned)

# Exercises:
# 1. Change the printed "Total due" in both PDFs to a wrong number (a vendor
#    mistake). Which check flags it? Your validation catches bad documents,
#    not just bad model output.
# 2. Increase the noise in make_scanned_pdf() (more speckles, stronger blur,
#    bigger rotation) until the vision route starts failing. Do the business
#    rules catch the failures, or do some wrong values still pass?
# 3. Install Tesseract (`sudo apt install tesseract-ocr`, `pip install
#    pytesseract`) and use pytesseract.image_to_string() as the OCR step, then
#    extract_from_text() on its output. Compare accuracy and cost with the
#    vision route.
