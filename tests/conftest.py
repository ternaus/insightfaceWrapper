from pathlib import Path

import cv2
import numpy as np
import torch

TARGET_IMAGE_SIZE = 112


def normalize(image: np.ndarray) -> np.ndarray:
    image /= 255
    image -= 0.5
    image /= 0.5
    return image


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(np.transpose(normalize(image), (2, 0, 1)))
    return torch.from_numpy(np.expand_dims(image, 0))


def prepare_image(image_path: Path) -> torch.Tensor:
    img = cv2.imread(image_path.as_posix())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    return tensor_from_rgb_image(img)


image_paths = [
    Path(__file__).parent / "data" / tx for tx in ("avenir1.jpg", "evdokia1.jpg", "natalia1.jpg", "avenir2.jpg")
]

embeddings = {
    "avenir1": [
        -0.02249542,
        -0.01845406,
        -0.0645938,
        -0.00631843,
        -0.03476263,
        -0.01797351,
        -0.01003856,
        -0.00600145,
        0.04645787,
        -0.03182485,
    ],
    "evdokia1": [
        0.06354304,
        -0.136894,
        0.02347769,
        0.03237138,
        -0.00858555,
        -0.0557017,
        0.01879028,
        0.03427035,
        -0.00873045,
        -0.02998946,
    ],
    "natalia1": [
        -0.04079512,
        -0.04061444,
        -0.10172332,
        0.01384232,
        -0.00199641,
        0.07662791,
        0.05439072,
        -0.0168717,
        -0.02620804,
        0.02527176,
    ],
    "avenir2": [
        0.04216152,
        -0.02330013,
        -0.00386324,
        0.00124177,
        -0.0658131,
        -0.01154281,
        0.04056456,
        -0.00638489,
        0.0866634,
        -0.01571165,
    ],
}

images = {
    image_path.stem: {
        "image": prepare_image(image_path),
        "image_path": image_path,
        "embedding": embeddings[image_path.stem],
    }
    for image_path in image_paths
}
