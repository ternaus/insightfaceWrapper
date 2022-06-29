import numpy as np
import torch
from pytest import mark

from insightfacewrapper.get_model import get_model
from tests.conftest import images

embedding_extractor = get_model("ms1mv3_arcface_r34")
embedding_extractor.eval()


@mark.parametrize(
    ["image", "embedding"], [(images[name]["image"], images[name]["embedding"]) for name in images.keys()]
)
def test_face_detection(image, embedding):
    assert image.shape == (1, 3, 112, 112)
    with torch.inference_mode():
        predicted_embedding = embedding_extractor(image).cpu().numpy()[0]

    assert np.allclose(embedding, predicted_embedding[: len(embedding)], rtol=1e-3)


@mark.parametrize("image", [images[name]["image"] for name in images.keys()])
def test_normalized(image):
    assert image.shape == (1, 3, 112, 112)

    with torch.inference_mode():
        predicted_embedding = embedding_extractor(image).cpu().numpy()[0]

    assert np.allclose([np.linalg.norm(predicted_embedding)], [1])


def test_closeness():
    predicted_embeddings = {}

    cos75 = 0.2588190451

    for key, value in images.items():
        with torch.inference_mode():
            predicted_embeddings[key] = embedding_extractor(value["image"]).cpu().numpy()[0]

    assert (predicted_embeddings["avenir1"] * predicted_embeddings["avenir2"]).sum() > cos75
    assert (predicted_embeddings["avenir1"] * predicted_embeddings["natalia1"]).sum() < cos75
    assert (predicted_embeddings["avenir1"] * predicted_embeddings["evdokia1"]).sum() < cos75
    assert (predicted_embeddings["avenir2"] * predicted_embeddings["natalia1"]).sum() < cos75
    assert (predicted_embeddings["avenir2"] * predicted_embeddings["evdokia1"]).sum() < cos75
    assert (predicted_embeddings["natalia1"] * predicted_embeddings["evdokia1"]).sum() < cos75
