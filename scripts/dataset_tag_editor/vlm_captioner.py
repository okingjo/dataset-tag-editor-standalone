"""Natural-language captioning via an OpenAI-compatible vision-language API.

Designed for local servers (Ollama / llama.cpp / vLLM) running an uncensored
VLM such as Qwen3.5-35B-A3B (abliterated), so NSFW anime datasets can be
captioned without refusals. Only ``requests`` + Pillow are needed; the model
itself runs on the server, not inside this app.

Output is stored per image in a ``<img>.nl.txt`` sidecar (see multi_caption.py)
and can be combined with the edited tag ``.txt`` into ``.multi`` candidates.
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert image captioner for anime-style illustration LoRA "
    "training. Write accurate, detailed natural-language captions. Describe "
    "only what is actually visible; never invent details. NSFW content is "
    "allowed and must be described factually and without refusal when present."
)

DEFAULT_PROMPT = (
    "Describe this image in one fluent paragraph of natural English prose for "
    "image-generation training. Cover: number of subjects, appearance (hair, "
    "eyes, body), clothing and accessories, pose and action, expression, "
    "background and setting, lighting, composition and art style. Be specific "
    "and factual. No markdown, no bullet lists, no preamble such as 'This "
    "image shows'. If existing tags are provided, use them as hints but verify "
    "them against the image."
)


def encode_image_data_uri(img_path: str | Path, max_side: int = 1024) -> str:
    """Downscale to max_side and encode as a JPEG data URI (RGB, no alpha)."""
    from PIL import Image

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        if max(im.size) > max_side:
            scale = max_side / max(im.size)
            im = im.resize(
                (max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_payload(
    img_path: str | Path,
    *,
    model: str,
    prompt: str = "",
    system_prompt: str = "",
    tags_hint: list[str] | None = None,
    enable_thinking: bool = False,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    image_max_side: int = 1024,
) -> dict:
    user_text = (prompt or DEFAULT_PROMPT).strip()
    if tags_hint:
        user_text += "\n\nExisting tags: " + ", ".join(t for t in tags_hint if t)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": encode_image_data_uri(img_path, image_max_side)}},
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        "temperature": temperature,
        # Qwen3.5 official non-thinking sampling: temp 0.7 / top_p 0.8 / top_k 20.
        # Ollama, llama.cpp and vLLM all accept top_k in the request body;
        # servers that don't know it ignore the field.
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": max_tokens,
        # Hybrid reasoning models (Qwen3/Qwen3.5): thinking improves caption
        # quality but is slow; make it explicit. Ollama reads "think", HF-style
        # servers read chat_template_kwargs; others ignore both.
        "think": bool(enable_thinking),
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }


_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "</" + "think" + ">"
_THINK_RE = re.compile(re.escape(_THINK_OPEN) + r".*?" + re.escape(_THINK_CLOSE), re.S)
_THINK_UNCLOSED_RE = re.compile(re.escape(_THINK_OPEN) + r".*$", re.S)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```$", re.S)


def clean_reply(text: str) -> str:
    text = _THINK_RE.sub("", text or "")
    text = _THINK_UNCLOSED_RE.sub("", text).strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    # collapse to a single paragraph (one .multi candidate = one line)
    return re.sub(r"\s+", " ", text).strip()


def caption_image(
    img_path: str | Path,
    *,
    base_url: str,
    model: str,
    api_key: str = "",
    prompt: str = "",
    system_prompt: str = "",
    tags_hint: list[str] | None = None,
    enable_thinking: bool = False,
    timeout: float = 300.0,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    image_max_side: int = 1024,
) -> str:
    """Caption one image. Raises RuntimeError with a readable message on failure."""
    import requests

    if not base_url or not model:
        raise RuntimeError("VLM captioning requires Base URL and Model (see Settings)")
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = build_payload(
        img_path,
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        tags_hint=tags_hint,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
        temperature=temperature,
        image_max_side=image_max_side,
    )
    try:
        resp = requests.post(base + "/chat/completions", headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"VLM request failed: {e}") from e
    if resp.status_code != 200:
        raise RuntimeError(f"VLM server HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected VLM response: {resp.text[:300]}") from e
    text = clean_reply(content)
    if not text:
        raise RuntimeError("VLM returned an empty caption (refusal or thinking-only output)")
    return text


def caption_images(
    img_paths: list[str],
    *,
    skip_existing: bool = True,
    tags_hint_getter=None,
    progress_cb=None,
    **options,
):
    """Batch generator: yields (img_path, status, text).

    status: 'ok' | 'skipped' | 'error'. Writes <img>.nl.txt on success.
    progress_cb(done, total, path) is called after each image if given.
    """
    from . import multi_caption as mc

    total = len(img_paths)
    for i, path in enumerate(img_paths):
        text = ""
        if skip_existing and mc.nl_sidecar_path(path).is_file():
            status = "skipped"
        else:
            try:
                hint = tags_hint_getter(path) if tags_hint_getter else None
                text = caption_image(path, tags_hint=hint, **options)
                mc.write_nl(path, text)
                status = "ok"
            except Exception as e:
                status = "error"
                text = str(e)
        if progress_cb:
            progress_cb(i + 1, total, path)
        yield path, status, text
