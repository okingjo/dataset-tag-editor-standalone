# Dataset Tag Editor Standalone

A WebUI tool to edit image caption datasets for Text2Image model training
(LoRA / DreamBooth / finetuning). Edit booru tags and natural-language
captions side by side, and export the right caption format for every trainer.

> Independently maintained fork of
> [toshiaki1729/dataset-tag-editor-standalone](https://github.com/toshiaki1729/dataset-tag-editor-standalone)
> (upstream inactive since July 2024).

![](pic/ss01.png)

The UI follows the training-data workflow as four tabs:
**1. Captioning** (load images → interrogate tags → generate NL captions) →
**2. Tag Editor** (filter & edit tags) → **3. Export Multi** (combine into
training-ready captions) → **Settings**.

## New features in this fork

### 1. Modern taggers: PixAI Tagger v1.0 + WD v3

Old interrogators (WD v1/v1.4/v2, DeepDanbooru, BLIP, BLIP2, GIT) were
removed. What remains is the current state of the art:

- **[PixAI Tagger v1.0](https://huggingface.co/pixai-labs/pixai-tagger-v1.0)**
  (ONNX) — 30,877 tags across general / character / copyright / style / meta /
  rating categories, with the official per-category thresholds applied
  internally (character 0.27, style 0.15, copyright 0.24, meta 0.17, rating
  argmax). The UI slider tunes the general category (default 0.17); rating
  tags are gated by *Use Rating tags* in Settings. Runs from the
  DirectML-validated export
  [Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8](https://huggingface.co/Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8)
  (fp16 on GPU, fp32 on CPU; ~947 MiB download on first use), with batched
  inference.
- **SmilingWolf's [WD Tagger](https://huggingface.co/SmilingWolf) v3 family**
  (`wd-vit-tagger-v3`, `wd-convnext-tagger-v3`, `wd-swinv2-tagger-v3`) and the
  large timm variants (`wd-vit-large-v14-tagger`, `wd-eva02-large-tagger-v3`).
- [Z3D-E621-Convnext](https://huggingface.co/toynya/Z3D-E621-Convnext) for
  e621-style datasets.
- Custom taggers can be added in `userscripts/taggers` (aesthetic score
  predictors are included).

**GPU acceleration:** the default `onnxruntime` wheel is CPU-only (~11 s/image
for PixAI). Swap the package and the taggers automatically prefer CUDA /
DirectML:

```sh
# Windows (any DX12 GPU)
uv pip uninstall onnxruntime && uv pip install onnxruntime-directml
# Linux + CUDA
uv pip uninstall onnxruntime && uv pip install onnxruntime-gpu
```

(`uv sync` restores the CPU wheel from `pyproject.toml` — re-apply after syncing.)

### 2. Natural-language captioning with an uncensored VLM

The **Captioning** tab generates descriptive natural-language
captions with any OpenAI-compatible **vision** server — Ollama, llama.cpp
(`llama-server`) or vLLM. Built for NSFW datasets: point it at an
uncensored/abliterated **Qwen3.5-35B-A3B** and captions are not refused
(verified model recipes [below](#serving-the-nl-captioner-verified-recipes)).

- Existing tags can be sent as hints; thinking mode is off by default for
  clean single-paragraph output; sampling follows Qwen3.5's official
  non-thinking settings (temp 0.7 / top_p 0.8 / top_k 20).
- Results are saved as per-image sidecars `<img>.nl.txt`, reviewable and
  editable in *Tag Editor → Edit Caption of Selected Image → VLM NL Caption*.
- **Per-folder context (`nl_prompt.txt`)**: drop a text file named
  `nl_prompt.txt` into any dataset folder to give the VLM authoritative
  background for every image inside it — ideal for character LoRAs organized
  by costume:

  ```
  dataset/miku/nl_prompt.txt                 All images show the character hatsune miku.
  dataset/miku/school_uniform/nl_prompt.txt  She is wearing a sailor school uniform.
  dataset/miku/swimsuit/nl_prompt.txt        She is wearing a blue one-piece swimsuit.
  ```

  With recursive loading the files are inherited from the dataset root down
  to each image's folder (outer first), so the character folder states the
  character once and each costume subfolder adds its own description. The
  context is woven into every generated caption automatically.

### 3. One master caption set for every trainer: `.multi` export

CLIP-based trainers (SD 1.5 / SDXL / Illustrious / Pony) want comma-separated
booru tags; DiT trainers with LLM text encoders (Anima, MOSS) want natural
language. This fork maintains **one** master caption set and derives both:

```
img.png
img.txt       <- pure tags (edited in this app; used as-is by SD/Illustrious/Pony)
img.nl.txt    <- natural-language caption (VLM-generated or hand-written)
img.multi     <- derived candidates for kohya sd-scripts multi-caption training
```

*Export Multi → Generate .multi* combines the edited `.txt` tags with the NL
sidecars into complete candidates separated by `---===---` (sd-scripts
samples one per training step):

```
masterpiece, best quality, score_7, safe, 1girl, long hair, school uniform, classroom
---===---
masterpiece, best quality, score_7, safe, 1girl, long hair, school uniform, classroom. A girl with long hair stands in a sunlit classroom, wearing a navy school uniform with a red ribbon.
---===---
A girl with long hair stands in a sunlit classroom, wearing a navy school uniform with a red ribbon. Soft afternoon light falls across the desks behind her.
```

Options: which candidates to include (tags only / tags+NL / NL only), tag
style (spaces for Anima/MOSS, underscores for raw booru; `score_*` keeps
underscores), and a quality prefix. Re-run any time after editing — `.multi`
is always re-derivable from `.txt` + `.nl.txt`.

Training-side notes (kohya sd-scripts):

- Use `caption_extension = ".multi"` for Anima/MOSS. A dataset cannot load
  `.txt` and `.multi` simultaneously — keep a pure-tag candidate (the default)
  to retain tag-style training signal.
- Anima: do **not** cache text-encoder outputs (candidates are sampled
  randomly per step; a cache would freeze one). MOSS: TE caching is fine.
- SD/Illustrious/Pony keep using the plain `.txt` files unchanged.

### 4. Smart tag deduplication

Drop generic tags when a more specific one is present (`white thighhighs`
removes `thighhighs`, `rose` removes `flower`/`flowers`). Two methods:

- **Rules (offline)** — built-in implication table + `<color> <noun>`
  heuristic, extendable with a custom JSON file (Settings → *Smart Dedup
  custom rules JSON file*). No network needed.
- **LLM API** — any OpenAI-compatible or Anthropic chat endpoint decides;
  identical tag sets are cached so each unique caption is sent only once.

Preview first, then apply to filtered images. In *Batch Edit Captions →
Smart Dedup*.

### 5. Booru tag autocompletion

[a1111-sd-webui-tagcomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete)-compatible
completion in every caption textbox, on by default: category colours, post
counts, alias search (`high_res` → `highres`), ↑/↓ + Enter/Tab to insert,
Esc to dismiss. Uses the bundled `tags/danbooru.csv` or any tagcomplete-format
CSV (e.g. `e621.csv`). Fully local — no network access. Tunable in Settings.

### 6. uv-only isolated install + large-dataset performance

All dependencies are pinned in `pyproject.toml` / `uv.lock`; uv downloads a
private Python 3.11 and keeps **everything** (packages, models, temp files)
inside this folder under `runtime/`. No system Python pollution. Gallery and
filtering paths were optimized for large datasets.

## Quick start

The only prerequisite is [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
git clone https://github.com/okingjo/dataset-tag-editor-standalone.git
cd dataset-tag-editor-standalone
```

- **Windows:** double-click `launch_uv.bat`
- **Linux / macOS:** `./webui_uv.sh`

First run creates `.venv/` and installs the pinned dependencies automatically
(CPU-only PyTorch by default — lightweight, works everywhere). For NVIDIA GPU
PyTorch:

```sh
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv run --no-sync scripts/launch.py   # --no-sync keeps the CUDA build
```

Google Colab:

```sh
%cd /content
!git clone https://github.com/okingjo/dataset-tag-editor-standalone.git
%cd /content/dataset-tag-editor-standalone
!pip install uv
!uv sync
!uv run scripts/launch.py --share
```

## Serving the NL captioner (verified recipes)

The base model [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
is natively multimodal (vision built in — no separate "VL" variant), 35B
total / 3B active MoE. Uncensored options that keep vision working:

- **Ollama (easiest):** `ollama pull huihui_ai/qwen3.5-abliterated:35b`
  (Q4_K_M, ~24 GB, text+image; needs Ollama ≥ 0.17.5). Higher quality:
  `huihui_ai/qwen3.5-abliterated:35b-a3b-q8_0` (~39 GB). Set *NL Caption VLM
  base URL* to `http://<host>:11434/v1`, model `huihui_ai/qwen3.5-abliterated:35b`
  (the app default). Higher-quality custom import (auto-split across GPUs;
  the two `FROM` lines load model + vision projector):
  ```bash
  hf download mradermacher/Huihui-Qwen3.5-35B-A3B-abliterated-GGUF \
    Huihui-Qwen3.5-35B-A3B-abliterated.Q6_K.gguf \
    Huihui-Qwen3.5-35B-A3B-abliterated.mmproj-f16.gguf --local-dir ./q35
  printf 'FROM ./q35/Huihui-Qwen3.5-35B-A3B-abliterated.Q6_K.gguf\nFROM ./q35/Huihui-Qwen3.5-35B-A3B-abliterated.mmproj-f16.gguf\nPARAMETER num_ctx 32768\n' > Modelfile
  ollama create qwen3.5-35b-abliterated -f Modelfile
  ```
  The long silent "gathering model components" phase is SHA-256 hashing of
  20–30 GB, not a hang.
- **llama.cpp (best community reputation for NSFW captioning):**
  [HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive)
  GGUF — download the main quant **and** the `mmproj-…-f16.gguf` vision
  projector (required for image input), then:
  ```sh
  llama-server -m Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
    --mmproj mmproj-Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf \
    --jinja -c 8192 --port 8080
  ```
  Point the app at `http://<host>:8080/v1`. Q4_K_M ≈ 21 GB + 0.9 GB mmproj
  fits a 24 GB GPU; Q8_0 ≈ 37 GB suits 48 GB+ or unified-memory boxes.
  Shortcut on recent builds (auto-downloads the matching mmproj):
  `llama-server -hf HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive:Q6_K --jinja -c 8192 --port 8080`.
  If image requests 500 with "image input is not supported", pass `--mmproj`
  explicitly. Needs a build with qwen3_5_moe / Gated-DeltaNet support.
- **vLLM:** serve `huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated` (BF16, ~67 GB).

Notes: keep thinking mode off for batch captioning (app default); avoid
temperature 0 with quantized Qwen3-family models (repetition loops).
Abliteration removes refusals, but community reports say the huihui build's
NSFW *visual vocabulary* can be shallow — the HauhauCS Aggressive build is
preferred for explicit datasets. Test a few samples before batch-running.

## Basic usage

1. **1. Captioning** — point at your image folder and Load; interrogators
   (PixAI / WD v3) can tag on load. Then generate NL captions with the VLM
   (per-folder `nl_prompt.txt` context applies automatically). Captions in
   filenames can be loaded; edited captions are saved as `.txt` (or kohya
   sd-scripts metadata JSON).
2. **2. Tag Editor** — *Filter by Tags* (AND/OR logic) or *Filter by
   Selection* (manual pick, Enter to add); all batch edits apply **only to
   filtered images**. *Batch Edit Captions*: append / replace / remove tags,
   sort, regex search & replace, Smart Dedup. *Edit Caption of Selected
   Image*: per-image editing, re-interrogate, VLM NL caption review.
3. **3. Export Multi** — render `.multi` candidates from the edited tags +
   NL sidecars.
4. **Save all changes** (top bar, available on every tab).

## Troubleshooting

- **No images shown, no error** — register the dataset directory in Settings
  → *Path whitelist to show images* (e.g. `C:\` on Windows), or enable
  temporary image files.
- **"All files must contained within the Gradio python app working
  directory"** (≤ v0.0.5) — Settings → set *Directory to save temporary
  files* and check *Force using temporary file*.
- **Laggy gallery with many/large images** — Settings → set a non-zero
  *Maximum resolution of …* to use smaller thumbnails.
- **PyTorch without CUDA** — the default install is CPU-only; see the CUDA
  swap in [Quick start](#quick-start) and launch with `uv run --no-sync`.
