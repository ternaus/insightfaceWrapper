import json
import os
from pathlib import Path

import torch.jit
from torch import nn
from torch.hub import download_url_to_file

with open(Path(__file__).parent / "model_list.json", encoding="UTF-8") as f:
    models = json.load(f)


def get_model(model_name: str) -> nn.Module:
    cache_path = Path("~/.torch/models").expanduser().absolute()
    cache_path.mkdir(exist_ok=True, parents=True)

    weight_file_path = cache_path / model_name

    if not os.path.exists(weight_file_path):
        download_url_to_file(models[model_name], weight_file_path.as_posix(), progress=True)

    return torch.jit.load(weight_file_path.as_posix()).float()
