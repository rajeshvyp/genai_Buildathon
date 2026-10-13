
from pathlib import Path
import hashlib
import re
import csv
from datetime import datetime

import pymupdf
import pytesseract
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

# Safe output location: does not overwrite data/processed.
OUTPUT_DIR = ROOT / "data" / "processed_review"
MANIFEST_PATH = OUTPUT_DIR / "processing_manifest_review.csv"

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

OCR_DPI = 350
OCR_LANGUAGES = "eng+tam"
OCR_TAMIL_LANGUAGE = "tam"
MIN_EMBEDDED_CHARACTERS = 30

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


# ============================================================
# BASIC UTILITIES
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def make_output_name(path, digest):
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem)
    return f"{safe_name}_{digest[:10]}.txt"


def clean_text(text):
    """Clean obvious formatting noise without deleting useful content."""
    if not text:
        return ""

    text = text.replace("\x00", " ").replace("\ufeff", "")
    lines = []

    for line in text.splitlines():
        line = line.strip()

        if re.search(r"file:///.*\.mht", line, re.IGNORECASE):
            continue

        if line.lower() in {
            "skip to main content",
            "screen reader access",
            "accessibility options",
        }:
            continue

        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def tamil_count(text):
    return len(re.findall(r"[\u0B80-\u0BFF]", text or ""))


def replacement_count(text):
    return (text or "").count("\ufffd")


def suspicious_tamil_text(text):
    """Heuristic only; it cannot guarantee OCR or encoding correctness."""
    if not text or tamil_count(text) == 0:
        return False

    mojibake_markers = ("à®", "à¯", "â€", "Ã")

    if any(marker in text for marker in mojibake_markers):
        return True

    return replacement_count(text) > 0


def ocr_page(page, language):
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(OCR_DPI / 72, OCR_DPI / 72),
        alpha=False,
        colorspace=pymupdf.csRGB,
    )

    image = Image.frombytes(
        "RGB",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )

    result = pytesseract.image_to_string(
        image,
        lang=language,
        config="--psm 3",
    )

    return clean_text(result)


# ============================================================
# PDF EXTRACTION HELPERS
# ============================================================

def extract_positioned_text(page):
    """
    Extract text blocks in approximate reading order.
    This can help with layout, but cannot perfectly reconstruct
    every multi-column table.
    """
    blocks = page.get_text("blocks", sort=True)
    block_texts = []

    for block in blocks:
        # PyMuPDF text block tuple:
        # x0, y0, x1, y1, text, block_no, block_type
        if len(block) >= 5 and isinstance(block[4], str):
            text = clean_text(block[4])

            if text:
                block_texts.append(text)

    return clean_text("\n\n".join(block_texts))


def extract_tables(page):
    """
    Extract tables when supported by the installed PyMuPDF version.
    Returns table text and a count. Table detection may fail on
    borderless or visually complex tables, so normal text is retained.
    """
    table_texts = []

    try:
        finder = page.find_tables()
        tables = finder.tables

        for table_number, table in enumerate(tables, start=1):
            rows = table.extract()

            if not rows:
                continue

            formatted_rows = []

            for row in rows:
                if not row:
                    continue

                cells = []

                for cell in row:
                    cell_text = clean_text(str(cell or ""))
                    cells.append(cell_text)

                if any(cells):
                    formatted_rows.append(" | ".join(cells))

            if formatted_rows:
                table_texts.append(
                    f"[Detected table {table_number}]\n"
                    + "\n".join(formatted_rows)
                )

    except Exception:
        # Table detection is optional; keep the page text as fallback.
        return [], 0

    return table_texts, len(table_texts)


def page_needs_table_review(text, table_count):
    """
    Flag pages likely to contain tables or irregular extraction.
    This is a review hint, not a claim that extraction is incorrect.
    """
    if table_count > 0:
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) >= 12:
        short_lines = sum(1 for line in lines if len(line) <= 25)

        if short_lines / len(lines) >= 0.60:
            return True

    return False


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(path):
    warnings = []
    page_texts = []
    ocr_attempts = 0
    table_pages = 0

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            method = "embedded"

            # First try layout-aware positioned text blocks.
            embedded = extract_positioned_text(page)

            # Attempt table extraction independently. We keep the
            # original page text because table detection can be imperfect.
            table_texts, table_count = extract_tables(page)

            if table_count:
                table_pages += 1

            selected = embedded

            should_ocr = (
                len(embedded) < MIN_EMBEDDED_CHARACTERS
                or suspicious_tamil_text(embedded)
            )

            if should_ocr:
                ocr_attempts += 1

                try:
                    tamil = ocr_page(page, OCR_TAMIL_LANGUAGE)
                    mixed = ocr_page(page, OCR_LANGUAGES)

                    if tamil_count(tamil) > 0 and tamil_count(mixed) == 0:
                        candidate = tamil
                        candidate_method = "ocr_tamil"
                    else:
                        candidate = mixed
                        candidate_method = "ocr_eng_tam"

                    if not embedded or suspicious_tamil_text(embedded):
                        if candidate.strip():
                            selected = candidate
                            method = candidate_method
                        else:
                            warnings.append(
                                f"REVIEW_OCR_EMPTY_PAGE_{page_number}"
                            )

                except Exception as error:
                    warnings.append(
                        f"REVIEW_OCR_ERROR_PAGE_{page_number}_"
                        f"{type(error).__name__}"
                    )

            if not selected.strip():
                warnings.append(f"REVIEW_EMPTY_PAGE_{page_number}")

            if suspicious_tamil_text(selected):
                warnings.append(
                    f"REVIEW_SUSPICIOUS_TAMIL_PAGE_{page_number}"
                )

            if replacement_count(selected):
                warnings.append(
                    f"REVIEW_REPLACEMENT_CHARACTERS_PAGE_{page_number}"
                )

            # Add detected table rows separately. This gives retrieval
            # a table-oriented representation as well as page text.
            # The original text is preserved, not replaced.
            page_content = selected

            if table_texts:
                page_content += (
                    "\n\n[TABLE EXTRACTION - VERIFY AGAINST ORIGINAL PDF]\n"
                    + "\n\n".join(table_texts)
                )

            if page_needs_table_review(selected, table_count):
                warnings.append(
                    f"REVIEW_LAYOUT_OR_TABLE_PAGE_{page_number}"
                )

            page_texts.append(
                f"--- Page {page_number} | method={method} ---\n"
                f"{page_content}"
            )

    if table_pages:
        warnings.append(f"TABLES_DETECTED_ON_{table_pages}_PAGES")

    return "\n\n".join(page_texts).strip(), warnings, ocr_attempts


# ============================================================
# TXT EXTRACTION
# ============================================================

def extract_txt(path):
    warnings = []

    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp1252", errors="replace")

    text = clean_text(text)

    if not text:
        warnings.append("REVIEW_EMPTY_TXT")

    if replacement_count(text):
        warnings.append("REVIEW_REPLACEMENT_CHARACTERS")

    if suspicious_tamil_text(text):
        warnings.append("REVIEW_SUSPICIOUS_TAMIL_ENCODING")

    return text, warnings, 0


# ============================================================
# MAIN PROCESSING
# ============================================================

def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Source folder missing: {RAW_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = sorted(
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    print(f"Source files found: {len(sources)}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("Existing data/processed files will not be changed.")
    print("Original data/raw files will not be changed.")

    rows = []
    seen_hashes = {}

    for index, source in enumerate(sources, start=1):
        print(f"\n[{index}/{len(sources)}] {source.name}")

        digest = sha256_file(source)
        relative_source = source.relative_to(RAW_DIR).as_posix()
        output_name = make_output_name(source, digest)
        output_path = OUTPUT_DIR / output_name

        warnings = []
        ocr_attempts = 0
        duplicate_of = ""
        status = "SUCCESS"
        extracted = ""

        if digest in seen_hashes:
            status = "DUPLICATE"
            duplicate_of = seen_hashes[digest]
            warnings.append("EXACT_DUPLICATE")
        else:
            seen_hashes[digest] = relative_source

            try:
                if source.suffix.lower() == ".pdf":
                    extracted, warnings, ocr_attempts = extract_pdf(source)
                else:
                    extracted, warnings, ocr_attempts = extract_txt(source)

                if not extracted.strip():
                    status = "REVIEW_REQUIRED"
                    warnings.append("REVIEW_NO_EXTRACTED_TEXT")

                elif warnings:
                    status = "REVIEW_REQUIRED"

                if output_path.exists():
                    # Do not silently overwrite a prior output.
                    status = "REVIEW_REQUIRED"
                    warnings.append("OUTPUT_ALREADY_EXISTS_NOT_OVERWRITTEN")

                elif extracted.strip():
                    header = (
                        f"Source file: {relative_source}\n"
                        f"Source type: {source.suffix.lower()[1:]}\n"
                        f"Source SHA-256: {digest}\n\n"
                        "--- Extracted text ---\n\n"
                    )

                    output_path.write_text(
                        header + extracted + "\n",
                        encoding="utf-8",
                    )

                    print(f"  Saved: {output_name}")
                    print(f"  Characters: {len(extracted):,}")

                if warnings:
                    print(f"  Review flags: {len(warnings)}")

            except Exception as error:
                status = "FAILED"
                warnings.append(
                    f"PROCESSING_ERROR_{type(error).__name__}: {error}"
                )
                print(f"  ERROR: {error}")

        rows.append({
            "source_file": relative_source,
            "source_sha256": digest,
            "output_file": output_name if status != "DUPLICATE" else "",
            "status": status,
            "ocr_attempts": ocr_attempts,
            "warnings": " | ".join(warnings),
            "duplicate_of": duplicate_of,
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        })

    if rows:
        with MANIFEST_PATH.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

    print("\nProcessing complete.")
    print(f"Files examined: {len(sources)}")
    print(f"Manifest saved to: {MANIFEST_PATH}")
    print("Review the extracted text before building a vector index.")


if __name__ == "__main__":
    main()
