from __future__ import annotations

import io
import logging
from typing import Any, List, Union

import pdfplumber
import pypdfium2
import pytesseract

from .base import BaseExtractor, ExtractedPage

logger = logging.getLogger("neuroflow-pdf-extractor")


def format_markdown_table(table_rows: List[List[Any]]) -> str:
    """Format 2D table array into a Markdown table string."""
    if not table_rows or not table_rows[0]:
        return ""
    clean_rows = [
        [str(cell or "").strip().replace("\n", " ") for cell in row]
        for row in table_rows
    ]
    headers = clean_rows[0]
    sep = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for row in clean_rows[1:]:
        padded = row + [""] * (len(headers) - len(row))
        lines.append("| " + " | ".join(padded[:len(headers)]) + " |")
    return "\n".join(lines)


class PDFExtractor(BaseExtractor):
    """
    High-performance PDF extractor:
    - Uses pypdfium2 for digital text extraction.
    - Automatically detects scanned pages (< 50 chars) and falls back to pytesseract OCR (--psm 6).
    - Extracts structured tables as markdown with pdfplumber (content_type='table').
    - Preserves accurate page numbering in metadata.
    """

    async def extract(self, source: Union[str, bytes, io.BytesIO], **kwargs) -> List[ExtractedPage]:
        results: List[ExtractedPage] = []

        # Read into bytes if needed for multi-library compatibility
        if isinstance(source, io.BytesIO):
            pdf_bytes = source.getvalue()
        elif isinstance(source, bytes):
            pdf_bytes = source
        else:
            with open(source, "rb") as f:
                pdf_bytes = f.read()

        # 1. Extract digital text / OCR scanned pages via pypdfium2
        pdf_doc = pypdfium2.PdfDocument(pdf_bytes)
        total_pages = len(pdf_doc)

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = pdf_doc[page_idx]
            textpage = page.get_textpage()
            raw_text = textpage.get_text_range() or ""
            stripped_text = raw_text.strip()
            is_scanned = len(stripped_text) < 50

            if is_scanned:
                try:
                    # Rasterize page to high-res PIL image
                    pil_img = page.render(scale=2.0).to_pil()
                    ocr_text = pytesseract.image_to_string(pil_img, config="--psm 6")
                    final_text = ocr_text.strip()
                except Exception as err:
                    logger.warning(f"OCR failed for page {page_num}: {err}")
                    final_text = stripped_text
            else:
                final_text = stripped_text

            if final_text:
                results.append(
                    ExtractedPage(
                        page_number=page_num,
                        content=final_text,
                        content_type="text",
                        metadata={
                            "page_number": page_num,
                            "total_pages": total_pages,
                            "is_scanned": is_scanned,
                        },
                    )
                )

        # 2. Extract tables as markdown via pdfplumber
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as plumber_doc:
                for page_idx, plumber_page in enumerate(plumber_doc.pages):
                    page_num = page_idx + 1
                    tables = plumber_page.extract_tables() or []
                    for t_idx, table in enumerate(tables):
                        table_md = format_markdown_table(table)
                        if table_md.strip():
                            results.append(
                                ExtractedPage(
                                    page_number=page_num,
                                    content=table_md,
                                    content_type="table",
                                    metadata={
                                        "page_number": page_num,
                                        "table_index": t_idx,
                                        "total_pages": total_pages,
                                    },
                                )
                            )
        except Exception as err:
            logger.warning(f"pdfplumber table extraction failed: {err}")

        # Sort pages primarily by page_number
        results.sort(key=lambda p: (p.page_number, 0 if p.content_type == "text" else 1))
        return results
