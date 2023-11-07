# insightfaceWrapper
Wrapper for easier inference for insightface

## Install
```
pip install -U insightfacewrapper
```

## Models

* `ms1mv3_arcface_r18`
* `ms1mv3_arcface_r34`
* `ms1mv3_arcface_r50`
* `ms1mv3_arcface_r100`
* `glint360k_cosface_r18`
* `glint360k_cosface_r34`
* `glint360k_cosface_r50`
* `glint360k_cosface_r100`


```python
from insightfacewrapper.get_model import get_model
model = get_model(<model_name>)
model.eval()
```

### Inference

Based on the original
[inference script](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/inference.py),
image should be resized to `(112, 112)`.

```python
def normalize(image: np.ndarray) -> np.ndarray:
    image /= 255
    image -= 0.5
    image /= 0.5
    return image

def image2input(image: np.ndarray) -> np.ndarray:
    transposed = np.transpose(image, (2, 0, 1)).astype(np.float32)
    return torch.from_numpy(normalize(np.expand_dims(np.ascontiguousarray(transposed), 0)))

torch_input = image2input(image)

with torch.inference_engine():
    result = model(torch_input)[0].cpu().numpy()
```
