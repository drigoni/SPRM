import unittest
import os
from PIL import Image
import numpy as np
from model.dataset import get_image_embedding, load_image
import clip


class ImageEmbeddingTest(unittest.TestCase):
    def test__get_image_embedding(self):
        device = "cpu"

        model, preprocess = clip.load("RN101", jit=False)
        model = model.to(device).float()

        image_id = "1100214449"

        data_root = "data/flickr30k"
        image_file = f"{data_root}/flickr30k_images/{image_id}.jpg"

        image = load_image(image_file)
        boxes = np.array([[0, 0, 70, 200], [50, 50, 100, 100]])

        embs = get_image_embedding(image, boxes, model, preprocess, device=device)

        self.assertEqual(embs.shape, (2, 512))
