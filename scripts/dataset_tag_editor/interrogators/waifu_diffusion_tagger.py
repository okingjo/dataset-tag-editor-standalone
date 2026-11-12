from PIL import Image
import numpy as np
from typing import Tuple

import launch, utilities, settings, devices
from paths import paths

class WaifuDiffusionTagger:
    # brought from https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py and modified

    def __init__(self, model_name, model_filename="model.onnx", label_filename="selected_tags.csv"):
        self.MODEL_FILENAME = model_filename
        self.LABEL_FILENAME = label_filename
        self.MODEL_REPO = model_name
        self.model = None
        self.labels = []

    def load(self):
        import huggingface_hub

        if not self.model:
            path_model = huggingface_hub.hf_hub_download(
                self.MODEL_REPO, self.MODEL_FILENAME, cache_dir=paths.setting_model_path
            )
            if settings.current.interrogator_use_cpu:
                providers = ["CPUExecutionProvider"]
            else:
                providers = [
                    "CUDAExecutionProvider",
                    "DmlExecutionProvider",
                    "CPUExecutionProvider",
                ]

            # onnxruntime is a declared dependency (pyproject.toml), pre-installed
            # by uv. Never pip-install anything at runtime.
            if not launch.is_installed("onnxruntime"):
                raise RuntimeError(
                    "onnxruntime is not installed. Start the app with launch_uv.bat / "
                    "webui_uv.sh (or run `uv sync` first) to install the pinned dependencies."
                )
            import onnxruntime as ort

            print(f"Running ONNX on {ort.get_device()}")
            
            self.model = ort.InferenceSession(path_model, providers=providers)

        path_label = huggingface_hub.hf_hub_download(
            self.MODEL_REPO, self.LABEL_FILENAME
        )
        import pandas as pd

        self.labels = pd.read_csv(path_label)["name"].tolist()

    def unload(self):
        if not settings.current.interrogator_keep_in_memory:
            self.model = None
            devices.torch_gc()

    # brought from https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py and modified
    def apply(self, image: Image.Image):
        if not self.model:
            return []

        _, height, width, _ = self.model.get_inputs()[0].shape

        # the way to fill empty pixels is quite different from original one;
        # original: fill by white pixels
        # this: repeat the pixels on the edge
        image = utilities.resize(image, (width, height))
        image_np = np.array(image, dtype=np.float32)
        # PIL RGB to OpenCV BGR
        image_np = image_np[:, :, ::-1]
        image_np = np.expand_dims(image_np, 0)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        probs = self.model.run([label_name], {input_name: image_np})[0]
        labels: list[Tuple[str, float]] = list(zip(self.labels, probs[0].astype(float)))

        return labels
