"""PixAI Tagger v1.0 via ONNX (Mexes export, DirectML-validated).

Spec verified against pixai-labs/pixai-tagger-v1.0 (tagger_pipeline.py,
config.json) and Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8 (README,
selected_tags.csv, export script):

- input  "input":      float32 [batch, 3, 1008, 1008], RGB NCHW, dynamic batch
- output "prediction": float32 [batch, 30877], PROBABILITIES (sigmoid is
  already inside the graph -- do NOT apply sigmoid again)
- preprocess: alpha-composite over white -> RGB, scale by 1008/max(w,h)
  (truncate), bilinear resize, center-pad with BLACK to 1008x1008,
  normalize (x/255 - 0.5)/0.5  => [-1, 1].  NOT the WD14 0-255 BGR path.
- selected_tags.csv: tag_id,name,category,count; row order == output index.
  WD-style category ids: 0=general 1=style 3=copyright 4=character 5=meta
  9=rating.
- recommended per-category thresholds (macro-F1 operating points from the
  original config.json): general 0.17, character 0.27, style 0.15,
  copyright 0.24, meta 0.17, rating 0.41 (rating usually taken as argmax).
"""

import csv
from typing import Tuple

import numpy as np
from PIL import Image

import devices
import launch
import settings
from paths import paths

MODEL_REPO = "Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8"
INPUT_SIZE = 1008

CATEGORY_BY_ID = {0: "general", 1: "style", 3: "copyright", 4: "character", 5: "meta", 9: "rating"}
CATEGORY_THRESHOLDS = {
    "general": 0.17,
    "character": 0.27,
    "style": 0.15,
    "copyright": 0.24,
    "meta": 0.17,
    "rating": 0.41,
}


def preprocess(image: Image.Image, size: int = INPUT_SIZE) -> np.ndarray:
    """Rescale-and-pad exactly like the original RescalePadProcessor."""
    if image.mode != "RGB":
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        image = bg.convert("RGB")

    scale = size / max(image.size)
    # original truncates (int), not rounds
    w, h = int(image.width * scale), int(image.height * scale)
    w, h = max(w, 1), max(h, 1)
    image = image.resize((w, h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(image, ((size - w) // 2, (size - h) // 2))

    x = np.asarray(canvas, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return x.transpose(2, 0, 1)  # [3, size, size]


class PixAITagger:
    def __init__(self, model_repo: str = MODEL_REPO):
        self.MODEL_REPO = model_repo
        self.model = None
        self.tags: list[Tuple[str, str]] = []  # (name, category) in output order

    def load(self):
        import huggingface_hub

        if not self.model:
            # fp16 weights on GPU (fp32 needs ~3.2 GB VRAM); fp16 math is
            # SLOWER than fp32 on CPU, so CPU mode uses the fp32 export.
            if settings.current.interrogator_use_cpu:
                model_filename = "model.onnx"
                providers = ["CPUExecutionProvider"]
            else:
                model_filename = "model_fp16.onnx"
                providers = [
                    ("CUDAExecutionProvider", {}),
                    ("DmlExecutionProvider", {"device_id": devices.get_dml_device_id()}),
                    "CPUExecutionProvider",
                ]
            path_model = huggingface_hub.hf_hub_download(
                self.MODEL_REPO, model_filename, cache_dir=paths.setting_model_path
            )

            if not launch.is_installed("onnxruntime"):
                raise RuntimeError(
                    "onnxruntime is not installed. Start the app with launch_uv.bat / "
                    "webui_uv.sh (or run `uv sync` first) to install the pinned dependencies."
                )
            import onnxruntime as ort

            print(f"Running ONNX on {ort.get_device()}")

            # DirectML-validated session options (Mexes README)
            so = ort.SessionOptions()
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.model = ort.InferenceSession(path_model, so, providers=providers)

        path_label = huggingface_hub.hf_hub_download(
            self.MODEL_REPO, "selected_tags.csv", cache_dir=paths.setting_model_path
        )
        self.tags = []
        with open(path_label, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cat = CATEGORY_BY_ID.get(int(row["category"]))
                if cat is None:
                    raise ValueError(f"Unknown PixAI tag category id: {row['category']}")
                self.tags.append((row["name"], cat))

    def unload(self):
        if not settings.current.interrogator_keep_in_memory:
            self.model = None
            devices.torch_gc()

    def _run(self, batch: np.ndarray) -> np.ndarray:
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        probs = self.model.run([output_name], {input_name: batch})[0]
        if probs.shape[-1] != len(self.tags):
            raise RuntimeError(
                f"PixAI model output size {probs.shape[-1]} != tag list size {len(self.tags)}"
            )
        return probs

    def apply(self, image: Image.Image):
        """Returns [(tag_name, probability, category), ...] for one image."""
        if not self.model:
            return []
        batch = preprocess(image)[None]
        probs = self._run(batch)[0]
        return [(name, float(p), cat) for (name, cat), p in zip(self.tags, probs)]

    def apply_multi(self, images: list[Image.Image], batch_size: int = 4):
        """Yields one [(tag, prob, category), ...] list per image, in order."""
        if not self.model:
            return
        tensors = [preprocess(img) for img in images]
        for start in range(0, len(tensors), batch_size):
            chunk = tensors[start : start + batch_size]
            probs = self._run(np.stack(chunk, axis=0))
            for row in probs:
                yield [(name, float(p), cat) for (name, cat), p in zip(self.tags, row)]
