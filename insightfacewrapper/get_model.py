import os
from pathlib import Path

import torch.jit
from torch import nn
from torch.hub import download_url_to_file

common_path = "https://github.com/ternaus/insightfaceWrapper/releases/download"

models = {
    "ms1mv3_arcface_r18": f"{common_path}/v0.0.1/ms1mv3_arcface_r18.jit",
    "ms1mv3_arcface_r34": f"{common_path}/v0.0.2/ms1mv3_arcface_r34.jit",
    "ms1mv3_arcface_r50": f"{common_path}/v0.0.2/ms1mv3_arcface_r50.jit",
    "ms1mv3_arcface_r100": f"{common_path}/v0.0.2/ms1mv3_arcface_r100.jit",
    "glint360k_cosface_r18": f"{common_path}/v0.0.2/glint360k_cosface_r18.jit",
    "glint360k_cosface_r34": f"{common_path}/v0.0.2/glint360k_cosface_r34.jit",
    "glint360k_cosface_r50": f"{common_path}/v0.0.2/glint360k_cosface_r50.jit",
    "glint360k_cosface_r100": f"{common_path}/v0.0.2/glint360k_cosface_r100.jit",
}


def get_model(model_name: str) -> nn.Module:
    cache_path = Path("~/.torch/models").expanduser().absolute()
    cache_path.mkdir(exist_ok=True, parents=True)

    weight_file_path = cache_path / model_name

    if not os.path.exists(weight_file_path):
        download_url_to_file(models[model_name], weight_file_path.as_posix(), progress=True)

    return torch.jit.load(weight_file_path.as_posix())
