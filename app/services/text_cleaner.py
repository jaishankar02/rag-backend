from __future__ import annotations

import re
import unicodedata
from typing import Any


# ============================================================
# Text normalization
# ============================================================

CHAR_REPLACEMENTS = {
    # Ligatures
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",

    # Symbols and special characters
    "–": "-",
    "—": "-",
    "…": "...",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "«": '"',
    "»": '"',
    "‐": "-",
    "⁄": "/",
    "†": "*",
    "‡": "**",
    "•": "*",

    # Non-breaking spaces and special whitespace
    "\u00a0": " ",
    "\u2000": " ",
    "\u2001": " ",
    "\u2002": " ",
    "\u2003": " ",
    "\u2004": " ",
    "\u2005": " ",
    "\u2006": " ",
    "\u2007": " ",
    "\u2008": " ",
    "\u2009": " ",
    "\u200a": " ",
    "\u202f": " ",
    "\u205f": " ",
    "\u3000": " ",
}

LATEX_LINE_RE = re.compile(r"^\\[A-Za-z@]+")

LATEX_ENV_RE = re.compile(r"\\(begin|end)\{[^}]+\}")

EQUATION_RE = re.compile(r"(\$\$?.+\$\$?|\\\[.+\\\]|\\\(.+\\\))")

CODE_LINE_RE = re.compile(
    r"^\s*("
    r"def |class |import |from |return |"
    r"if |for |while |try:|except |with |"
    r"print\(|"
    r"[A-Za-z_][A-Za-z0-9_]*\s*="
    r")",
    re.IGNORECASE,
)

HEADER_FOOTER_RE = re.compile(r"^(?:page\s*)?\d+(?:\s*(?:/|of)\s*\d+)?$", re.IGNORECASE)

ROMAN_RE = re.compile(r"^(?:[ivxlcdm]+)$", re.IGNORECASE)

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

HYPHEN_LINEBREAK_RE = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")


def normalize_unicode(text: str) -> str:
    """Normalize Unicode, fix ligatures, accents, and special characters."""

    text = unicodedata.normalize("NFKC", text)

    for old, new in CHAR_REPLACEMENTS.items():
        text = text.replace(old, new)

    return text


def looks_like_latex(line: str) -> bool:
    """Detect LaTeX-like lines that should be preserved."""

    return bool(
        LATEX_LINE_RE.search(line)
        or LATEX_ENV_RE.search(line)
        or line.startswith(("\\", "{", "["))
    )


def looks_like_equation(line: str) -> bool:
    """Detect equation-like lines that should be preserved."""

    return bool(
        EQUATION_RE.search(line)
        or any(
            token in line
            for token in (
                "=",
                "^",
                "_",
                "\\frac",
                "\\sum",
                "\\alpha",
                "\\beta",
                "\\mu",
                "\\sigma",
            )
        )
    )


def looks_like_code(line: str) -> bool:
    """Detect code-like lines that should be preserved."""

    return bool(
        CODE_LINE_RE.search(line)
        or line.startswith(("    ", "\t"))
        or any(
            token in line
            for token in (
                ";",
                "::",
                "=>",
                "->",
                "{",
                "}",
                "def ",
                "class ",
            )
        )
    )


def looks_like_caption_or_heading(line: str) -> bool:
    """Detect headings and captions so they keep their wording and punctuation."""

    stripped = line.strip()

    if not stripped:
        return False

    if stripped.startswith(("Figure", "Table", "Fig.", "Caption", "Algorithm")):
        return True

    word_count = len(stripped.split())

    if (
        word_count <= 8
        and len(stripped) <= 80
        and stripped[:1].isupper()
        and not stripped.endswith(".")
        and not any(punct in stripped for punct in (",", ";", ":"))
    ):
        return True

    return False


def clean_normal_line(line: str) -> str:
    """Clean a regular prose line while keeping meaning intact."""

    line = re.sub(r"[ \f\v]+", " ", line.strip())

    line = re.sub(r"([!?.,;:])\1{2,}", r"\1\1", line)

    line = re.sub(r"([\-_=~*])\1{3,}", r"\1\1\1", line)

    line = re.sub(r"\s+([,.;:!?])", r"\1", line)

    return line


def _is_header_footer_candidate(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and (HEADER_FOOTER_RE.fullmatch(stripped) or ROMAN_RE.fullmatch(stripped)))


def clean_preserving_meaning(text: str) -> str:
    """
    Clean extracted text while preserving equations, code, LaTeX, headings,
    and captions.
    """

    if not text:
        return ""

    text = normalize_unicode(text)

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    text = HYPHEN_LINEBREAK_RE.sub("", text)
    text = CONTROL_CHARS_RE.sub(" ", text)

    cleaned_lines: list[str] = []
    prose_buffer: list[str] = []

    def flush_prose_buffer() -> None:
        if not prose_buffer:
            return

        paragraph = " ".join(prose_buffer)
        paragraph = clean_normal_line(paragraph)
        if paragraph:
            cleaned_lines.append(paragraph)
        prose_buffer.clear()

    previous_blank = False

    for raw_line in text.split("\n"):
        line = raw_line.rstrip()

        if not line.strip():
            previous_blank = True
            continue

        previous_blank = False
        stripped = line.strip()

        if _is_header_footer_candidate(stripped):
            flush_prose_buffer()
            continue

        if (
            looks_like_latex(line)
            or looks_like_equation(line)
            or looks_like_code(line)
            or looks_like_caption_or_heading(line)
        ):
            flush_prose_buffer()
            cleaned_lines.append(line.rstrip())
            continue

        prose_buffer.append(clean_normal_line(line))

    flush_prose_buffer()

    return "\n".join(cleaned_lines).strip()


def clean_text(text: str) -> str:
    """Backward-compatible wrapper for the main cleaner."""

    return clean_preserving_meaning(text)


def clean_block_text(text: str) -> str:
    """Alias for block-level extracted text cleaning."""

    return clean_preserving_meaning(text)


# ============================================================
# Page/block helpers
# ============================================================

def is_blank_text(text: str) -> bool:
    return not bool(text and text.strip())


def is_page_number_text(text: str) -> bool:
    stripped = text.strip() if text else ""
    return bool(stripped and (HEADER_FOOTER_RE.fullmatch(stripped) or ROMAN_RE.fullmatch(stripped)))


def is_ocr_artifact(text: str) -> bool:
    if not text:
        return True

    compact = re.sub(r"\s+", "", text).strip()
    if len(compact) <= 2:
        return True
    if len(set(compact)) == 1 and len(compact) >= 3:
        return True

    alpha_num = sum(ch.isalnum() for ch in compact)
    ratio = alpha_num / max(len(compact), 1)
    if ratio < 0.25 and len(compact) <= 12:
        return True

    return False


def is_header_footer_zone(block: dict[str, Any], page_height: float | None) -> bool:
    if not page_height:
        return False

    bbox = block.get("bbox") or []
    if len(bbox) != 4:
        return False

    y0 = bbox[1]
    y1 = bbox[3]
    top_zone = page_height * 0.12
    bottom_zone = page_height * 0.88
    return y0 <= top_zone or y1 >= bottom_zone


def clean_extracted_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clean extracted PDF pages and block text.

    The page-level filter removes obvious headers, footers, page numbers,
    blank text, and OCR garbage, while preserving the block structure.
    """

    if not pages:
        return []

    cleaned_pages: list[dict[str, Any]] = []

    # track repeated header/footer candidates across pages
    counts: dict[str, int] = {}

    for page in pages:
        page_height = page.get("page_height")
        cleaned_blocks: list[dict[str, Any]] = []

        for block in page.get("blocks", []):
            kind = block.get("kind")

            if kind != "text":
                cleaned_blocks.append(block)
                continue

            cleaned_text = clean_block_text(block.get("text", ""))
            if is_blank_text(cleaned_text) or is_ocr_artifact(cleaned_text):
                continue

            if is_page_number_text(cleaned_text):
                continue

            if len(cleaned_text) <= 80 and is_header_footer_zone(block, page_height):
                key = cleaned_text.lower()
                counts[key] = counts.get(key, 0) + 1

            cleaned_blocks.append({**block, "text": cleaned_text})

        cleaned_pages.append({**page, "blocks": cleaned_blocks})

    total_pages = len(pages)
    repeated_cutoff = max(2, total_pages // 2 + (total_pages % 2))

    final_pages: list[dict[str, Any]] = []
    for page in cleaned_pages:
        page_height = page.get("page_height")
        final_blocks: list[dict[str, Any]] = []

        for block in page.get("blocks", []):
            if block.get("kind") != "text":
                final_blocks.append(block)
                continue

            text = block.get("text", "")
            if is_blank_text(text) or is_ocr_artifact(text):
                continue

            if is_page_number_text(text):
                continue

            if len(text) <= 80 and is_header_footer_zone(block, page_height):
                if counts.get(text.lower(), 0) >= repeated_cutoff:
                    continue

            final_blocks.append(block)

        final_pages.append({**page, "blocks": final_blocks})

    return final_pages
