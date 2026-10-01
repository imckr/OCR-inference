"""Markdown to HTML conversion for OCR model output.

This implementation intentionally delegates parsing to markdown-it so we preserve
complex structure from PaddleOCR-style outputs: raw HTML blocks, markdown tables,
and LaTeX-style dollar delimiters.
"""

import re
from html import escape

from markdown_it import MarkdownIt


_md = (
    MarkdownIt(
        "commonmark",
        {
            # Preserve model-emitted raw HTML blocks such as <div>, <img>, and <table>.
            "html": True,
            # Keep paragraph semantics; do not force every newline into <br>.
            "breaks": False,
        },
    )
    .enable("table")
)


_CELL_STYLE = "text-align: center; word-wrap: break-word;"
_TABLE_CHUNK_RE = re.compile(r"((?:(?:<fcel>|<ecel>).*?<nl>\s*){2,})", re.DOTALL)


def _decode_paddle_table_tokens(text: str) -> str:
    """Convert PaddleOCR table tokens to HTML table markup.

    Some outputs emit table structure as token streams:
    `<fcel>` (filled cell), `<ecel>` (empty cell), and `<nl>` (row delimiter).
    """

    if "<fcel>" not in text and "<ecel>" not in text:
        return text

    def _chunk_to_table(match: re.Match[str]) -> str:
        chunk = match.group(1)
        rows: list[list[str]] = []

        for raw_row in chunk.split("<nl>"):
            row = raw_row.strip()
            if not row:
                continue

            cells: list[str] = []
            i = 0
            while i < len(row):
                if row.startswith("<ecel>", i):
                    cells.append("")
                    i += len("<ecel>")
                    continue

                if row.startswith("<fcel>", i):
                    i += len("<fcel>")
                    next_filled = row.find("<fcel>", i)
                    next_empty = row.find("<ecel>", i)
                    token_positions = [p for p in (next_filled, next_empty) if p != -1]
                    cell_end = min(token_positions) if token_positions else len(row)
                    cells.append(row[i:cell_end].strip())
                    i = cell_end
                    continue

                i += 1

            if cells:
                rows.append(cells)

        if not rows:
            return chunk

        max_cols = max(len(r) for r in rows)
        html_rows: list[str] = []
        for r in rows:
            padded = r + [""] * (max_cols - len(r))
            cells_html = "".join(
                f"<td style='{_CELL_STYLE}'>{escape(cell)}</td>" for cell in padded
            )
            html_rows.append(f"<tr>{cells_html}</tr>")

        return (
            "<table border=1 style='margin: auto; word-wrap: break-word;'>"
            + "".join(html_rows)
            + "</table>"
        )

    return _TABLE_CHUNK_RE.sub(_chunk_to_table, text)


# ── LaTeX delimiter & operatorname normalisation ──────────────────────
_DISPLAY_MATH_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
_OPERATORNAME_RE = re.compile(r"\\operatorname\{([^}]+)\}")


def _collapse_operatorname(match: re.Match[str]) -> str:
    inner = match.group(1).replace(" ", "")
    return rf"\operatorname{{{inner}}}"


def _normalize_latex(text: str) -> str:
    r"""Normalise LaTeX delimiters and spacing quirks.

    * ``\[ ... \]``  →  ``$$ ... $$``
    * ``\( ... \)``  →  ``$ ... $``
    * ``\operatorname{A t t e n t i o n}``  →  ``\operatorname{Attention}``
    """
    text = _OPERATORNAME_RE.sub(_collapse_operatorname, text)
    text = _DISPLAY_MATH_RE.sub(r" $$ \1 $$ ", text)
    text = _INLINE_MATH_RE.sub(r" $ \1 $ ", text)
    return text


def normalize_model_output(text: str) -> str:
    """Full post-processing pipeline for raw model output."""
    text = _decode_paddle_table_tokens(text)
    text = _normalize_latex(text)
    return text


def md_to_html(text: str) -> str:
    """Render markdown to HTML without flattening structure."""
    if not text or not text.strip():
        return ""

    text = normalize_model_output(text)

    return _md.render(text)
