import unittest

import clip

from model.dataset import get_text_embedding


class TextEmbeddingTest(unittest.TestCase):
    def test__get_text_embedding(self):
        device = "cpu"

        model, _ = clip.load("RN101", jit=False)
        model = model.to(device).float()

        queries = ["left man", "black and white dog"]

        text_embedding = get_text_embedding(
            queries, model, clip.tokenize, device=device
        )

        self.assertEqual(text_embedding.shape, (2, 512))
