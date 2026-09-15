from typing import List, Dict, Any, Optional
import os
from pathlib import Path


def _save_image_bytes(image_bytes: bytes, output_dir: str, file_name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, file_name)
    with open(save_path, "wb") as f:
        f.write(image_bytes)
    return save_path


def _extract_block_text(block: Dict[str, Any]) -> str:
    block_text_parts: List[str] = []
    if "lines" in block:
        for line in block["lines"]:
            for span in line.get("spans", []):
                txt = span.get("text", "")
                if txt:
                    block_text_parts.append(txt)
    else:
        maybe_text = block.get("text") or ""
        if maybe_text:
            block_text_parts.append(maybe_text)
    return "\n".join(block_text_parts).strip()


def _build_image_entry(
    *,
    block: Dict[str, Any],
    doc: Any,
    pdf_stem: str,
    page_number: int,
    image_index: int,
    output_dir: str,
) -> Optional[Dict[str, Any]]:
    bbox = block.get("bbox", [])
    xref = block.get("xref")

    image_bytes = block.get("image")
    image_info: Dict[str, Any] = {}

    if image_bytes is None and xref:
        try:
            image_info = doc.extract_image(xref)
            image_bytes = image_info.get("image")
        except Exception:
            image_bytes = None

    if image_bytes is None:
        return None

    ext = (block.get("ext") or image_info.get("ext") or "png").lower()
    file_name = f"{pdf_stem}_p{page_number}_img{image_index}_{xref or 'na'}.{ext}"
    image_path = _save_image_bytes(image_bytes, output_dir, file_name)

    return {
        "kind": "image",
        "type": block.get("type"),
        "bbox": bbox,
        "image_path": image_path,
        "meta": {
            "xref": xref,
            "width": block.get("width", image_info.get("width")),
            "height": block.get("height", image_info.get("height")),
            "colorspace": block.get("colorspace", image_info.get("colorspace")),
            "bpc": block.get("bpc", image_info.get("bpc")),
            "ext": ext,
            "size": block.get("size", image_info.get("size")),
            "name": block.get("name"),
            "transform": block.get("transform"),
        },
    }


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract PDF content page-by-page.

    Output is page-based and includes both text and image blocks.
    - Text blocks keep their extracted text.
    - Image blocks are saved to disk and store `image_path` instead of text.
    - Bounding boxes, block type, and metadata are preserved.
    """
    try:
        import fitz  # PyMuPDF
    except Exception:
        raise RuntimeError(
            "PyMuPDF is required for PDF extraction. Install with `pip install pymupdf` "
            "and add `pymupdf` to your requirements.txt."
        )

    doc = fitz.open(pdf_path)
    pdf_stem = Path(pdf_path).stem or "document"
    output_dir = os.path.join(Path(pdf_path).resolve().parent.parent, "uploads", "images")

    pages_out: List[Dict[str, Any]] = []

    for page_index, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict")
        blocks_out: List[Dict[str, Any]] = []
        image_index = 0

        for block in page_dict.get("blocks", []):
            bbox = block.get("bbox", [])
            block_type = block.get("type")

            if block_type == 1:
                image_index += 1
                image_entry = _build_image_entry(
                    block=block,
                    doc=doc,
                    pdf_stem=pdf_stem,
                    page_number=page_index,
                    image_index=image_index,
                    output_dir=output_dir,
                )
                if image_entry is not None:
                    blocks_out.append(image_entry)
                continue

            text_value = _extract_block_text(block)
            blocks_out.append(
                {
                    "kind": "text",
                    "type": block_type,
                    "bbox": bbox,
                    "text": text_value,
                    "meta": {
                        "number": block.get("number"),
                        "lines": len(block.get("lines", [])),
                    },
                }
            )

        pages_out.append(
            {
                "page_number": page_index,
                "page_width": page.rect.width,
                "page_height": page.rect.height,
                "blocks": blocks_out,
            }
        )

    doc.close()
    return pages_out
