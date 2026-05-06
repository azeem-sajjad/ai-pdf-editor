import re
import json
import fitz
import faiss
import ollama
import numpy as np
from io import BytesIO
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# ── Embedding model (loads once) ───────────────────────────────────────────────
_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# ── PDF Extraction ─────────────────────────────────────────────────────────────

def extract_text_and_blocks(pdf_bytes: bytes) -> tuple:
    """
    Extract full text and per-span metadata (position, font, size) from every page.
    This metadata is what lets us edit inline without destroying formatting.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_blocks = {}
    all_page_texts = []

    for page_num, page in enumerate(doc):
        spans_data = []
        page_lines = []
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_parts = []
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    if span_text.strip():
                        spans_data.append({
                            "text":  span_text,
                            "bbox":  span["bbox"],
                            "font":  span.get("font", "helv"),
                            "size":  round(span.get("size", 11.0), 1),
                            "color": span.get("color", 0),
                            "page":  page_num,
                        })
                    line_parts.append(span_text)
                page_lines.append("".join(line_parts))

        page_blocks[page_num] = spans_data
        all_page_texts.append("\n".join(page_lines))

    doc.close()
    full_text = "\n\n".join(all_page_texts)
    return full_text, page_blocks


# ── Chunking & FAISS Index ─────────────────────────────────────────────────────

def _split_sentences(text: str) -> list:
    """Split text into sentences without needing NLTK."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'])", text)
    return [p.strip() for p in parts if p.strip()]


def create_index(text: str, chunk_sentences: int = 6, overlap: int = 2):
    """
    Build a FAISS vector index from sentence-aware chunks.
    Overlap of 2 sentences prevents context from getting lost at chunk edges.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return [], None

    step = max(1, chunk_sentences - overlap)
    chunks = []
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i: i + chunk_sentences]).strip()
        if len(chunk) > 40:
            chunks.append(chunk)

    if not chunks:
        return [], None

    model = get_embedding_model()
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return chunks, index


def search(query: str, chunks: list, index) -> list:
    """Return the top-5 most relevant chunks for a query."""
    if not chunks or index is None:
        return []
    model = get_embedding_model()
    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    k = min(5, len(chunks))
    _, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]


# ── LLM — Q&A via Ollama (streaming) ──────────────────────────────────────────

def ask_llm_stream(query: str, context: list, model_name: str = "llama3.2"):
    """
    Stream an answer grounded in retrieved context chunks.
    Yields text deltas for real-time display in Streamlit.
    """
    context_block = "\n\n---\n\n".join(context)
    prompt = f"""You are a precise document assistant.
Answer the question using ONLY the document context below.
If the answer is not in the context, say: "This information is not in the document."
Be concise and cite specific details.

DOCUMENT CONTEXT:
{context_block}

QUESTION:
{query}

ANSWER:"""

    stream = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk["message"]["content"]
        if delta:
            yield delta


# ── LLM — Structured Edits (JSON find-replace list) ───────────────────────────

def get_structured_edits(original_text: str, instruction: str, model_name: str = "llama3.2") -> list:
    """
    Ask the LLM to return a JSON array of find/replace pairs.
    These are applied surgically to the PDF without touching anything else.
    """
    prompt = f"""You are a document editor. Output ONLY a valid JSON array, no explanation, no markdown.
Each item must be: {{"find": "exact text from document", "replace": "new text"}}
"find" must be an EXACT substring from the document (one sentence or short phrase).

INSTRUCTION: {instruction}

DOCUMENT:
{original_text[:6000]}

Return ONLY the JSON array:"""

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response["message"]["content"].strip()

    # Strip any markdown fences the model might add
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw).strip()

    # Extract just the JSON array if there's surrounding text
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        edits = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Model returned invalid JSON.\n\nRaw output:\n{raw}\n\n"
            "Tip: Try switching to Full Rewrite mode."
        )

    return [e for e in edits if "find" in e and "replace" in e]


# ── LLM — Full Document Rewrite ────────────────────────────────────────────────

def rewrite_full_document(original_text: str, instruction: str, model_name: str = "llama3.2") -> str:
    """Complete document rewrite. Returns the full edited text."""
    prompt = f"""You are a professional document editor.
Return ONLY the complete edited document. No commentary, no preamble, no labels.
Preserve paragraph structure and line breaks.

INSTRUCTION: {instruction}

DOCUMENT:
{original_text}

EDITED DOCUMENT:"""

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


# ── PDF Output — Precise inline edit (PyMuPDF redaction) ──────────────────────

def _get_font_size_for_text(page: fitz.Page, search_text: str) -> float:
    """Find the font size used by a piece of text on a given page."""
    needle = search_text[:30]
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if needle in span.get("text", ""):
                    return round(span.get("size", 11.0), 1)
    return 11.0


def apply_changes_to_pdf(pdf_bytes: bytes, changes: list) -> bytes:
    """
    Apply find-replace changes to the PDF using PyMuPDF redaction.

    For each change:
      1. Find all matching text rectangles on every page
      2. White-out those rectangles (redact)
      3. Insert the replacement text at the exact same position
         using the original font size

    Everything else — images, tables, other text, page layout — stays untouched.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for change in changes:
        find_text  = change.get("find", "").strip()
        replace_text = change.get("replace", "").strip()

        if not find_text or find_text == replace_text:
            continue

        for page in doc:
            instances = page.search_for(find_text)
            if not instances:
                continue

            font_size = _get_font_size_for_text(page, find_text)

            # Phase 1 — mark all instances
            for rect in instances:
                page.add_redact_annot(rect, fill=(1, 1, 1))

            # Phase 2 — apply all redactions at once
            page.apply_redactions()

            # Phase 3 — insert replacement text at original positions
            for rect in instances:
                page.insert_text(
                    fitz.Point(rect.x0, rect.y1),
                    replace_text,
                    fontsize=font_size,
                    color=(0, 0, 0),
                )

    buf = BytesIO()
    doc.save(buf, garbage=4, deflate=True)
    doc.close()
    return buf.getvalue()


# ── PDF Output — Full Rewrite (styled ReportLab) ──────────────────────────────

def create_styled_pdf(text: str) -> bytes:
    """Build a clean, professionally styled PDF from plain text."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )

    base = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "Body", parent=base["Normal"],
        fontSize=11, leading=17, spaceAfter=6,
    )
    heading_style = ParagraphStyle(
        "Head", parent=base["Heading2"],
        fontSize=14, leading=20,
        spaceBefore=12, spaceAfter=6,
        textColor=colors.HexColor("#1a1a2e"),
    )

    story = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 8))
            continue

        is_heading = (
            len(stripped) < 80
            and (stripped.isupper() or stripped.rstrip().endswith(":"))
        )
        style = heading_style if is_heading else body_style
        safe = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        try:
            story.append(Paragraph(safe, style))
        except Exception:
            story.append(Paragraph(re.sub(r"[<>&]", " ", stripped), body_style))

    doc.build(story)
    return buf.getvalue()


# ── Diff ───────────────────────────────────────────────────────────────────────

def compute_diff_html(original: str, edited: str) -> str:
    """Word-level diff: green = added, red strikethrough = removed."""
    orig_words = original.split()
    edit_words = edited.split()
    matcher = SequenceMatcher(None, orig_words, edit_words, autojunk=False)

    ADD = "background:#d4edda;color:#155724;border-radius:3px;padding:1px 5px"
    DEL = "background:#f8d7da;color:#721c24;text-decoration:line-through;border-radius:3px;padding:1px 5px"

    parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(orig_words[i1:i2]))
        elif tag == "insert":
            parts.append(f'<mark style="{ADD}"> {" ".join(edit_words[j1:j2])} </mark>')
        elif tag == "delete":
            parts.append(f'<mark style="{DEL}"> {" ".join(orig_words[i1:i2])} </mark>')
        elif tag == "replace":
            parts.append(f'<mark style="{DEL}"> {" ".join(orig_words[i1:i2])} </mark>')
            parts.append(f'<mark style="{ADD}"> {" ".join(edit_words[j1:j2])} </mark>')
        parts.append(" ")

    return (
        '<div style="font-family:system-ui,sans-serif;font-size:14px;'
        'line-height:2;white-space:pre-wrap">' + "".join(parts) + "</div>"
    )