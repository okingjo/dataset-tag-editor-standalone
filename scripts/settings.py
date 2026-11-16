import json
from typing import NamedTuple
from utilities import base_dir_path

SETTING_PATH = base_dir_path() / "settings.json"

class Settings(NamedTuple):
    allowed_paths: str = ''
    use_temp_files: bool = False
    temp_directory: str = ''
    cleanup_tmpdir: bool = True
    image_columns: int = 6
    max_resolution: int = 0
    filename_word_regex: str = ""
    filename_join_string: str = " "
    tagger_use_spaces: bool = True
    interrogator_use_cpu: bool = False
    interrogator_keep_in_memory: bool = False
    interrogator_model_dir: str = ""
    tagger_use_rating: bool = False
    num_cpu_worker: int = -1
    # [PERF] gallery pagination: max number of images rendered per page (0 = all)
    gallery_page_size: int = 200
    # Smart tag deduplication: optional LLM backend (OpenAI-compatible or Anthropic)
    dedup_llm_api_type: str = "openai"
    dedup_llm_base_url: str = ""
    dedup_llm_model: str = ""
    dedup_llm_api_key: str = ""
    dedup_rules_file: str = ""
    # Booru tag autocompletion (a1111-sd-webui-tagcomplete compatible)
    tagcomplete_enabled: bool = True
    tagcomplete_tag_file: str = ""
    tagcomplete_max_results: int = 20
    tagcomplete_search_alias: bool = True
    tagcomplete_replace_underscores: bool = True
    tagcomplete_append_comma: bool = True
    batch_size_vit:int = 4
    batch_size_vit_large:int = 4
    batch_size_convnext:int = 4
    batch_size_swinv2:int = 4
    batch_size_eva02_large:int = 4
    batch_size_pixai:int = 4
    # Natural-language captioning via an OpenAI-compatible VLM server
    # (Ollama / llama.cpp / vLLM), e.g. an uncensored Qwen3.5-35B-A3B.
    caption_llm_base_url: str = "http://127.0.0.1:11434/v1"
    caption_llm_model: str = "huihui_ai/qwen3.5-abliterated:35b"
    caption_llm_api_key: str = ""
    caption_llm_timeout: float = 300.0
    caption_llm_enable_thinking: bool = False
    caption_llm_max_tokens: int = 2048
    caption_llm_image_size: int = 1024
    caption_llm_prompt: str = ""
    caption_llm_system_prompt: str = ""


DEFAULT = Settings()
current = Settings()


NAMES = list(Settings.__annotations__.keys())

DESCRIPTIONS = {
    "allowed_paths": "Path whitelist to show images in gallery (for local use). Split paths by comma and space (, )",
    "use_temp_files": "Force using temporary file to show images on gallery",
    "temp_directory": "Directory to save temporary files",
    "cleanup_tmpdir": "Cleanup temporary files on startup",
    "image_columns": "Column number of image galleries",
    "max_resolution": "Maximum resolution of gallery thumbnails (0 to disable)",
    "filename_word_regex": "regex to read caption from filename",
    "filename_join_string" : "Characters to insert between the words which matches with regex",
    "tagger_use_spaces": "Replace underbar (_) in tags with whitespace ( )",
    "interrogator_use_cpu": "Use CPU to interrogate",
    "interrogator_keep_in_memory": "Keep interroagor in VRAM",
    "interrogator_model_dir": "Path to directory for downloaded interrogator models",
    "tagger_use_rating": "Use Rating tags when using Tagger",
    "num_cpu_worker": "Number of CPU workers when loading or preprocessing images (set -1 to auto)",
    "gallery_page_size": "[PERF] Max images shown per gallery page (0 to show all; smaller = snappier UI)",
    "dedup_llm_api_type": "Smart Dedup LLM API type: 'openai' (OpenAI-compatible: llama.cpp / Ollama / vLLM / OpenAI) or 'anthropic'",
    "dedup_llm_base_url": "Smart Dedup LLM base URL, e.g. http://127.0.0.1:8080 or https://api.openai.com (with or without /v1)",
    "dedup_llm_model": "Smart Dedup LLM model name, e.g. gpt-4o-mini or a local model id",
    "dedup_llm_api_key": "Smart Dedup LLM API key (leave empty for local servers without auth)",
    "dedup_rules_file": "Smart Dedup custom rules JSON file (optional; extends the built-in implication table)",
    "tagcomplete_enabled": "Enable booru tag autocompletion while editing captions (a1111-sd-webui-tagcomplete compatible)",
    "tagcomplete_tag_file": "Tag autocomplete CSV file (tagcomplete format). Empty = bundled tags/danbooru.csv; point to your tagcomplete tags folder for e621/custom files",
    "tagcomplete_max_results": "Tag autocomplete: maximum number of suggestions shown",
    "tagcomplete_search_alias": "Tag autocomplete: also search tag aliases",
    "tagcomplete_replace_underscores": "Tag autocomplete: insert tags with spaces instead of underscores",
    "tagcomplete_append_comma": "Tag autocomplete: append a comma and space after inserting a tag",
    "batch_size_vit": "Inference batch size for ViT taggers",
    "batch_size_vit_large": "Inference batch size for ViT large taggers",
    "batch_size_convnext": "Inference batch size for ConvNeXt taggers",
    "batch_size_swinv2": "Inference batch size for SwinV2 taggers",
    "batch_size_eva02_large": "Inference batch size for EVA-02 large taggers",
    "batch_size_pixai": "Inference batch size for PixAI Tagger v1.0 (1008x1008 input; lower it if VRAM is tight)",
    "caption_llm_base_url": "NL Caption VLM base URL (OpenAI-compatible), e.g. http://127.0.0.1:11434/v1 (Ollama) or http://127.0.0.1:8080 (llama.cpp)",
    "caption_llm_model": "NL Caption VLM model name. Ollama: huihui_ai/qwen3.5-abliterated:35b (Q4_K_M, vision) or :35b-a3b-q8_0; llama.cpp: the model name you gave llama-server (e.g. HauhauCS Qwen3.5-35B-A3B Uncensored Aggressive GGUF + mmproj)",
    "caption_llm_api_key": "NL Caption VLM API key (leave empty for local servers without auth)",
    "caption_llm_timeout": "NL Caption VLM per-image request timeout in seconds",
    "caption_llm_enable_thinking": "NL Caption VLM: enable thinking mode (hybrid models only; slower but can improve quality)",
    "caption_llm_max_tokens": "NL Caption VLM: maximum tokens per caption",
    "caption_llm_image_size": "NL Caption VLM: longest image side sent to the server (px)",
    "caption_llm_prompt": "NL Caption VLM: custom user prompt (empty = built-in anime captioning prompt)",
    "caption_llm_system_prompt": "NL Caption VLM: custom system prompt (empty = built-in uncensored captioner prompt)",
}



def save():
    SETTING_PATH.write_text(json.dumps(current._asdict()), "utf8")


def load():
    global current
    if SETTING_PATH.is_file():
        settings = DEFAULT._asdict() | json.loads(SETTING_PATH.read_text("utf8"))
    else:
        settings = DEFAULT._asdict()
    settings = {key:value for key,value in settings.items() if key in DEFAULT._asdict()}
    current = Settings(**settings)


def restore_defaults():
    global current
    current = Settings()