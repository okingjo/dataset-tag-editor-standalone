# Only the modern WD v3 family is kept. The legacy v1.4 v1/v2 taggers,
# DeepDanbooru and the BLIP/BLIP2/GIT captioners were removed (superseded by
# WD v3 / PixAI for tags and the VLM captioner for natural language).
WD_TAGGERS = {
    "wd-vit-tagger-v3": 0.35,
    "wd-convnext-tagger-v3": 0.35,
    "wd-swinv2-tagger-v3": 0.35,
}

WD_TAGGERS_TIMM = {
    "wd-vit-large-v14-tagger": 0.35,
    "wd-eva02-large-tagger-v3": 0.35,
}

# PixAI Tagger v1.0 (ONNX). The value is the GENERAL-category threshold;
# other categories use the official per-category operating points internally.
PIXAI_TAGGERS = {
    "pixai-tagger-v1.0": 0.17,
}
