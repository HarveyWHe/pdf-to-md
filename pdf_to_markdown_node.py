"""
PDF To Markdown Node - Convert a PDF document into LLM-friendly Markdown.

The node extracts:
- Body text (PyMuPDF block layout)
- Tables (pdfplumber, rendered as GitHub-Flavored-Markdown tables)
- Images (PyMuPDF, exported as PNG, optionally OCR'd with Tesseract)

Items on each page are merged in reading order (top-to-bottom, left-to-right)
so the resulting Markdown preserves the original document flow.

Required : PyMuPDF (`pymupdf`), pdfplumber
Optional : pytesseract + system `tesseract` binary (image OCR; degrades gracefully)
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import io
import json

import fitz  # PyMuPDF
import pdfplumber

try:
    import pytesseract
    from PIL import Image  # Pillow ships with pytesseract's deps in practice
    _PYTESSERACT_IMPORTED = True
except ImportError:
    pytesseract = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    _PYTESSERACT_IMPORTED = False


BBox = Tuple[float, float, float, float]


class PDFToMarkdownNode:
    """
    Convert a PDF document into a Markdown file optimized for LLM consumption.

    The Markdown contains the document title (PDF stem), one ``## Page N``
    section per page, with text paragraphs, tables, and image references
    interleaved in reading order. Images are written as PNG files into a
    sibling ``images/`` directory and referenced via standard Markdown image
    syntax. When Tesseract OCR is available, image text is also embedded as
    an HTML comment beneath each image so text-only LLMs can use it.
    """

    DEFAULT_OUTPUT_DIR = "output_md"
    DEFAULT_IMAGE_SUBDIR = "images"
    DEFAULT_OCR_LANG = "eng"

    TABLE_STRICT_SETTINGS: Dict[str, Any] = {
        "vertical_strategy": "lines_strict",
        "horizontal_strategy": "lines_strict",
        "snap_tolerance": 5,
        "join_tolerance": 5,
        "edge_min_length": 3,
    }
    CONTINUATION_TOP_RATIO = 0.30
    CONTINUATION_BOTTOM_RATIO = 0.70

    def __init__(self, pdf_path: Optional[str] = None):
        self._doc: Optional[fitz.Document] = None
        self._pdf_path: Optional[str] = None
        self.last_result: Optional[Dict[str, Any]] = None
        if pdf_path:
            self.load(pdf_path)

    # ============ Loading ============

    def load(self, pdf_path: str) -> "PDFToMarkdownNode":
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            raise RuntimeError(f"Failed to open PDF '{pdf_path}': {exc}") from exc

        if doc.is_encrypted:
            # Try empty password (some PDFs are encrypted but not password-protected).
            if not doc.authenticate(""):
                doc.close()
                raise RuntimeError(
                    f"PDF '{pdf_path}' is password-protected; this node does not "
                    "support password input. Decrypt the file first."
                )

        self._doc = doc
        self._pdf_path = str(path)
        return self

    # ============ Extraction ============

    def extract(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        image_subdir: str = DEFAULT_IMAGE_SUBDIR,
        ocr: bool = True,
        ocr_lang: str = DEFAULT_OCR_LANG,
    ) -> Dict[str, Any]:
        """
        Extract the loaded PDF into Markdown plus image assets.

        Parameters
        ----------
        output_dir : str
            Directory where the ``.md`` file and ``images/`` folder are written.
        image_subdir : str
            Subdirectory (inside ``output_dir``) where image files are stored.
        ocr : bool
            Attempt OCR on each extracted image. Silently disabled if neither
            ``pytesseract`` nor a working ``tesseract`` binary is available.
        ocr_lang : str
            Tesseract language code(s), e.g. ``"eng"`` or ``"eng+chi_sim"``.

        Returns
        -------
        dict
            Summary including ``markdown_file``, per-page stats, and metadata.
        """
        if not self._doc or not self._pdf_path:
            raise RuntimeError("No PDF loaded. Call load() first.")

        ocr_enabled = bool(ocr) and self._is_ocr_runtime_available()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        img_dir = out / image_subdir

        stem = Path(self._pdf_path).stem
        md_path = out / f"{stem}.md"

        page_summaries: List[Dict[str, Any]] = []
        total_tables = 0
        total_images = 0

        md_chunks: List[str] = [f"# {stem}\n"]

        with pdfplumber.open(self._pdf_path) as plumber_doc:
            page_count = min(len(self._doc), len(plumber_doc.pages))

            page_items_list: List[List[Dict[str, Any]]] = []
            page_heights: List[float] = []
            for page_index in range(page_count):
                page = self._doc[page_index]
                plumber_page = plumber_doc.pages[page_index]
                items, page_height = self._collect_page_items(
                    page=page,
                    plumber_page=plumber_page,
                    page_index=page_index,
                    image_dir=img_dir,
                    image_rel_subdir=image_subdir,
                    ocr_enabled=ocr_enabled,
                    ocr_lang=ocr_lang,
                )
                page_items_list.append(items)
                page_heights.append(page_height)

            self._merge_cross_page_tables(page_items_list, page_heights)

            for page_index, items in enumerate(page_items_list):
                page_md, summary = self._render_items(items, page_index)
                md_chunks.append(f"## Page {page_index + 1}\n")
                md_chunks.append(page_md)
                page_summaries.append(summary)
                total_tables += summary["tables"]
                total_images += summary["images"]

        md_path.write_text("\n".join(md_chunks).rstrip() + "\n", encoding="utf-8")

        result: Dict[str, Any] = {
            "markdown_file": str(md_path),
            "metadata": {
                "source_pdf": self._pdf_path,
                "pages": len(self._doc),
                "tables": total_tables,
                "images": total_images,
                "ocr_enabled": ocr_enabled,
                "ocr_lang": ocr_lang if ocr_enabled else None,
            },
            "pages": page_summaries,
        }
        self.last_result = result
        return result

    # ============ Per-page collection / rendering ============

    def _collect_page_items(
        self,
        page: "fitz.Page",
        plumber_page,
        page_index: int,
        image_dir: Path,
        image_rel_subdir: str,
        ocr_enabled: bool,
        ocr_lang: str,
    ) -> Tuple[List[Dict[str, Any]], float]:
        items: List[Dict[str, Any]] = []
        page_height = float(getattr(page.rect, "height", 0.0) or 0.0)

        table_bboxes: List[BBox] = []
        for table in self._find_tables_with_fallback(plumber_page):
            try:
                data = table.extract()
            except Exception:
                continue
            if not data:
                continue
            bbox = self._safe_bbox(getattr(table, "bbox", None))
            if bbox is None:
                continue
            normalized = self._normalize_table(data)
            if not normalized:
                continue
            table_bboxes.append(bbox)
            items.append(
                {
                    "type": "table",
                    "y": bbox[1],
                    "x": bbox[0],
                    "y_bottom": bbox[3],
                    "data": normalized,
                }
            )

        try:
            image_info = page.get_image_info(xrefs=True)
        except Exception:
            image_info = []

        for img_idx, info in enumerate(image_info):
            xref = info.get("xref", 0)
            if not xref:
                continue
            bbox = self._safe_bbox(info.get("bbox"))
            if bbox is None:
                continue
            try:
                extracted = self._doc.extract_image(xref) if self._doc else None
            except Exception:
                extracted = None
            if not extracted:
                continue
            img_bytes = extracted.get("image")
            if not img_bytes:
                continue
            ext = (extracted.get("ext") or "png").lower()

            image_dir.mkdir(parents=True, exist_ok=True)
            filename = f"page{page_index + 1}_img{img_idx + 1}.{ext}"
            (image_dir / filename).write_bytes(img_bytes)

            ocr_text = ""
            if ocr_enabled:
                ocr_text = self._safe_ocr_image(img_bytes, ocr_lang)

            items.append(
                {
                    "type": "image",
                    "y": bbox[1],
                    "x": bbox[0],
                    "filename": filename,
                    "rel_path": f"{image_rel_subdir}/{filename}",
                    "ocr_text": ocr_text,
                }
            )

        try:
            text_dict = page.get_text("dict")
        except Exception:
            text_dict = {"blocks": []}

        for block in text_dict.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            bbox = self._safe_bbox(block.get("bbox"))
            if bbox is None:
                continue
            if self._inside_any(bbox, table_bboxes):
                continue
            text = self._extract_block_text(block)
            if not text:
                continue
            items.append(
                {
                    "type": "text",
                    "y": bbox[1],
                    "x": bbox[0],
                    "text": text,
                }
            )

        items.sort(key=lambda it: (round(it["y"], 1), round(it["x"], 1)))
        return items, page_height

    def _render_items(
        self,
        items: List[Dict[str, Any]],
        page_index: int,
    ) -> Tuple[str, Dict[str, Any]]:
        out_lines: List[str] = []
        char_count = 0
        table_count = 0
        image_count = 0

        for item in items:
            kind = item["type"]
            if kind == "text":
                out_lines.append(item["text"])
                out_lines.append("")
                char_count += len(item["text"])
            elif kind == "table":
                table_md = self._table_to_markdown(item["data"])
                if table_md:
                    out_lines.append(table_md)
                    out_lines.append("")
                    table_count += 1
            elif kind == "image":
                alt = self._first_line_alt(item["ocr_text"]) or "image"
                out_lines.append(f"![{alt}]({item['rel_path']})")
                if item["ocr_text"]:
                    sanitized = item["ocr_text"].replace("-->", "--&gt;")
                    out_lines.append("")
                    out_lines.append(f"<!-- OCR:\n{sanitized}\n-->")
                out_lines.append("")
                image_count += 1

        page_md = "\n".join(out_lines).rstrip() + "\n"

        summary = {
            "index": page_index + 1,
            "chars": char_count,
            "tables": table_count,
            "images": image_count,
        }
        return page_md, summary

    # ============ Cross-page table merging ============

    def _merge_cross_page_tables(
        self,
        page_items_list: List[List[Dict[str, Any]]],
        page_heights: List[float],
    ) -> None:
        """
        Merge tables that visually continue across page breaks.

        A table on page N is treated as a continuation of the trailing table of
        the running chain when:

        - Both have the same column count after normalization.
        - The chain's last table extended to the bottom band of its page
          (y_bottom beyond ``CONTINUATION_BOTTOM_RATIO`` of page height).
        - The current table starts in the top band of its page (y_top before
          ``CONTINUATION_TOP_RATIO`` of page height).
        - There is no other content item above the current table on the new
          page.

        Continuation propagates across multiple pages: a long table that spans
        pages 1..4 is merged into one Markdown table.
        """
        chain_end: Optional[Dict[str, Any]] = None
        chain_end_page_height: float = 0.0

        for i, items in enumerate(page_items_list):
            if not items:
                continue
            height = page_heights[i] if i < len(page_heights) else 0.0
            if height <= 0:
                page_last = self._last_table(items)
                if page_last is not None:
                    chain_end = page_last
                    chain_end_page_height = height
                continue

            first_table = self._first_table(items)
            if (
                chain_end is not None
                and first_table is not None
                and not self._has_item_above(items, first_table)
                and chain_end_page_height > 0
                and chain_end["y_bottom"]
                >= chain_end_page_height * self.CONTINUATION_BOTTOM_RATIO
                and first_table["y"] <= height * self.CONTINUATION_TOP_RATIO
            ):
                prev_cols = max(
                    (len(r) for r in chain_end["data"]), default=0
                )
                cur_cols = max(
                    (len(r) for r in first_table["data"]), default=0
                )
                if prev_cols > 0 and prev_cols == cur_cols:
                    cur_rows = list(first_table["data"])
                    if (
                        cur_rows
                        and chain_end["data"]
                        and self._rows_equal(chain_end["data"][0], cur_rows[0])
                    ):
                        cur_rows = cur_rows[1:]

                    chain_end["data"] = list(chain_end["data"]) + cur_rows
                    chain_end["y_bottom"] = height
                    chain_end_page_height = height
                    items.remove(first_table)

            page_last = self._last_table(items)
            if page_last is not None:
                chain_end = page_last
                chain_end_page_height = height

    @staticmethod
    def _last_table(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        last = None
        for it in items:
            if it["type"] == "table":
                last = it
        return last

    @staticmethod
    def _first_table(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for it in items:
            if it["type"] == "table":
                return it
        return None

    @staticmethod
    def _has_item_above(
        items: List[Dict[str, Any]], target: Dict[str, Any]
    ) -> bool:
        target_y = target["y"]
        for it in items:
            if it is target:
                continue
            if it["y"] < target_y - 1:
                return True
        return False

    @staticmethod
    def _rows_equal(
        row_a: List[Optional[str]], row_b: List[Optional[str]]
    ) -> bool:
        normalize = lambda r: [
            (str(c).strip() if c is not None else "") for c in (r or [])
        ]
        return normalize(row_a) == normalize(row_b)

    # ============ Text helpers ============

    @staticmethod
    def _extract_block_text(block: Dict[str, Any]) -> str:
        lines_out: List[str] = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans)
            line_text = line_text.strip()
            if line_text:
                lines_out.append(line_text)
        return "\n".join(lines_out).strip()

    @staticmethod
    def _safe_bbox(value: Any) -> Optional[BBox]:
        if value is None:
            return None
        try:
            x0, y0, x1, y1 = (float(v) for v in value)
        except (TypeError, ValueError):
            return None
        return (x0, y0, x1, y1)

    @staticmethod
    def _inside_any(bbox: BBox, bboxes: Iterable[BBox]) -> bool:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        for b in bboxes:
            if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                return True
        return False

    # ============ Table helpers ============

    def _find_tables_with_fallback(self, plumber_page) -> List[Any]:
        """Try strict lines-based detection first; fall back to defaults."""
        try:
            strict = (
                plumber_page.find_tables(table_settings=self.TABLE_STRICT_SETTINGS)
                or []
            )
        except Exception:
            strict = []
        if strict:
            return strict
        try:
            return plumber_page.find_tables() or []
        except Exception:
            return []

    @staticmethod
    def _normalize_table(
        rows: List[List[Optional[str]]],
    ) -> List[List[Optional[str]]]:
        """Drop fully empty rows and pad each row to a consistent column count."""
        kept = [row for row in (rows or []) if any((c or "").strip() for c in (row or []))]
        if not kept:
            return []
        max_cols = max(len(row) for row in kept)
        return [list(row) + [""] * (max_cols - len(row)) for row in kept]

    def _table_to_markdown(self, rows: List[List[Optional[str]]]) -> str:
        normalized = self._normalize_table(rows)
        if not normalized:
            return ""

        compacted = self._drop_empty_columns(normalized)
        if not compacted:
            return ""

        cleaned = [[self._clean_cell(cell) for cell in row] for row in compacted]
        max_cols = len(cleaned[0])

        header = cleaned[0]
        body = cleaned[1:] if len(cleaned) > 1 else []

        if not any(cell.strip() for cell in header):
            header = [f"col{i + 1}" for i in range(max_cols)]
            body = cleaned

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * max_cols) + " |",
        ]
        for row in body:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    @staticmethod
    def _drop_empty_columns(
        rows: List[List[Optional[str]]],
    ) -> List[List[Optional[str]]]:
        if not rows:
            return rows
        n_cols = len(rows[0])
        keep = [
            c
            for c in range(n_cols)
            if any((row[c] or "").strip() for row in rows)
        ]
        if len(keep) == n_cols:
            return rows
        if not keep:
            return []
        return [[row[c] for c in keep] for row in rows]

    @staticmethod
    def _clean_cell(cell: Optional[str]) -> str:
        if cell is None:
            return ""
        text = str(cell).replace("\r", " ").replace("\n", " ").strip()
        return text.replace("|", "\\|")

    # ============ Image / OCR helpers ============

    @staticmethod
    def _first_line_alt(text: str) -> str:
        if not text:
            return ""
        first = text.strip().splitlines()[0] if text.strip() else ""
        first = first.replace("[", "(").replace("]", ")")
        return first[:80].strip()

    def _is_ocr_runtime_available(self) -> bool:
        if not _PYTESSERACT_IMPORTED:
            return False
        try:
            pytesseract.get_tesseract_version()  # type: ignore[union-attr]
        except Exception:
            return False
        return True

    @staticmethod
    def _safe_ocr_image(img_bytes: bytes, ocr_lang: str) -> str:
        if not _PYTESSERACT_IMPORTED:
            return ""
        try:
            with Image.open(io.BytesIO(img_bytes)) as pil_img:  # type: ignore[union-attr]
                text = pytesseract.image_to_string(pil_img, lang=ocr_lang)  # type: ignore[union-attr]
        except Exception:
            return ""
        return text.strip()

    # ============ JSON output (Capacity boundary) ============

    def to_json(self, result: Optional[Dict[str, Any]] = None) -> str:
        payload = result if result is not None else self.last_result
        return json.dumps(
            payload or {},
            indent=2,
            default=str,
            ensure_ascii=False,
        )


# ============ CLI entry point ============

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a PDF (text + tables + images) into LLM-friendly Markdown."
    )
    parser.add_argument("pdf_file", help="Path to the source PDF file")
    parser.add_argument(
        "-o", "--output-dir",
        default=PDFToMarkdownNode.DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {PDFToMarkdownNode.DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--image-subdir",
        default=PDFToMarkdownNode.DEFAULT_IMAGE_SUBDIR,
        help=(
            "Sub-directory under the output dir where image assets are written "
            f"(default: {PDFToMarkdownNode.DEFAULT_IMAGE_SUBDIR})"
        ),
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip OCR even if Tesseract is available",
    )
    parser.add_argument(
        "--ocr-lang",
        default=PDFToMarkdownNode.DEFAULT_OCR_LANG,
        help=(
            "Tesseract language code(s) for OCR, e.g. 'eng' or 'eng+chi_sim' "
            f"(default: {PDFToMarkdownNode.DEFAULT_OCR_LANG})"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also write a JSON summary alongside the .md file and print it to stdout",
    )
    args = parser.parse_args()

    node = PDFToMarkdownNode(args.pdf_file)
    result = node.extract(
        output_dir=args.output_dir,
        image_subdir=args.image_subdir,
        ocr=not args.no_ocr,
        ocr_lang=args.ocr_lang,
    )

    md = result["markdown_file"]
    meta = result["metadata"]
    print(
        f"\nDone. {meta['pages']} page(s), {meta['tables']} table(s), "
        f"{meta['images']} image(s) -> '{md}'."
    )
    if not meta["ocr_enabled"]:
        print("OCR disabled (Tesseract not available or --no-ocr passed).")

    if args.json:
        json_str = node.to_json(result)
        json_path = Path(md).with_suffix(".json")
        json_path.write_text(json_str, encoding="utf-8")
        print(json_str)


if __name__ == "__main__":
    main()
