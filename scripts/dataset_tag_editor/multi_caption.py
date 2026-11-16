"""Hybrid caption export for DiT trainers (Anima / MOSS) and SD-family trainers.

One master caption set, two derived artifacts per image:

* ``<img>.txt``   -- pure booru tags, untouched. Used as-is by SD / Illustrious /
                     Pony (CLIP) training.
* ``<img>.multi`` -- caption candidates separated by ``---===---`` lines, used by
                     kohya sd-scripts' multi-caption support (Anima / MOSS).
                     Each block is a COMPLETE caption; the trainer randomly picks
                     one per step. Typical candidates: pure tags / tags+NL hybrid
                     / pure natural language.

Natural-language captions live in a sidecar ``<img>.nl.txt`` (written by the VLM
captioner or by hand). The ``.multi`` file is always re-derivable from
``.txt`` (or the editor's in-memory tags) + ``.nl.txt``, so only the master data
needs maintenance.
"""

from __future__ import annotations

import re
from pathlib import Path

MULTI_SEPARATOR = "---===---"

# Tags whose underscores are semantic and must survive space conversion
# (Anima quality/score tags: score_7, score_8, ...).
_KEEP_UNDERSCORE = re.compile(r"^score_\d+$", re.IGNORECASE)


def nl_sidecar_path(img_path: str | Path) -> Path:
    return Path(img_path).with_suffix(".nl.txt")


def multi_sidecar_path(img_path: str | Path, ext: str = ".multi") -> Path:
    ext = ext if ext.startswith(".") else "." + ext
    return Path(img_path).with_suffix(ext)


def read_nl(img_path: str | Path) -> str:
    p = nl_sidecar_path(img_path)
    if p.is_file():
        return p.read_text("utf8").strip()
    return ""


def write_nl(img_path: str | Path, text: str) -> Path:
    p = nl_sidecar_path(img_path)
    p.write_text(text.strip() + "\n", "utf8")
    return p


def read_tags_from_txt(img_path: str | Path, caption_ext: str = ".txt") -> list[str]:
    """Read the saved tag caption file (the 'edited txt' source of truth)."""
    p = Path(img_path).with_suffix(caption_ext)
    if not p.is_file():
        return []
    text = p.read_text("utf8")
    return [t.strip() for t in text.split(",") if t.strip()]


def _tag_to_spaces(tag: str) -> str:
    if _KEEP_UNDERSCORE.match(tag):
        return tag
    return tag.replace("_", " ")


def _tag_to_underscores(tag: str) -> str:
    # collapse whitespace runs; keep already-underscored tags intact
    return re.sub(r"\s+", "_", tag.strip())


def format_tag_line(
    tags: list[str],
    style: str = "spaces",
    quality_prefix: str = "",
) -> str:
    """Render tags as one comma-separated line.

    style="spaces"      -> Anima/MOSS friendly (long hair, school uniform)
    style="underscores" -> raw booru style (long_hair, school_uniform)
    quality_prefix is a comma-separated string prepended verbatim in the same
    style (e.g. "masterpiece, best quality, score_7, safe").
    """
    conv = _tag_to_spaces if style == "spaces" else _tag_to_underscores
    parts: list[str] = []
    if quality_prefix.strip():
        parts.extend(conv(t.strip()) for t in quality_prefix.split(",") if t.strip())
    parts.extend(conv(t.strip()) for t in tags if t and t.strip())
    # dedupe while preserving order (prefix may repeat a tag)
    seen: set[str] = set()
    out: list[str] = []
    for t in parts:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return ", ".join(out)


def _normalize_nl(text: str) -> str:
    # one line per candidate: collapse all whitespace (incl. newlines)
    return re.sub(r"\s+", " ", text).strip()


def build_candidates(
    tags: list[str],
    nl_text: str,
    *,
    include_tags: bool = True,
    include_hybrid: bool = True,
    include_nl: bool = True,
    tag_style: str = "spaces",
    quality_prefix: str = "",
    prefix_nl: bool = False,
) -> list[str]:
    """Build the ordered candidate list for one image's .multi file."""
    tag_line = format_tag_line(tags, tag_style, quality_prefix)
    nl = _normalize_nl(nl_text)
    candidates: list[str] = []
    if include_tags and tag_line:
        candidates.append(tag_line)
    if include_hybrid and tag_line and nl:
        candidates.append(tag_line + ". " + nl)
    if include_nl and nl:
        if prefix_nl and quality_prefix.strip():
            prefix = format_tag_line([], tag_style, quality_prefix)
            candidates.append(prefix + ". " + nl)
        else:
            candidates.append(nl)
    # dedupe identical candidates, keep order
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def compose_multi(candidates: list[str], separator: str = MULTI_SEPARATOR) -> str:
    return ("\n" + separator + "\n").join(candidates) + "\n"


def write_multi(
    img_path: str | Path,
    candidates: list[str],
    ext: str = ".multi",
    overwrite: bool = True,
) -> str:
    """Write <img>.multi. Returns 'written' | 'exists' | 'empty'."""
    if not candidates:
        return "empty"
    p = multi_sidecar_path(img_path, ext)
    if p.is_file() and not overwrite:
        return "exists"
    p.write_text(compose_multi(candidates), "utf8")
    return "written"
