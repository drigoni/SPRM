import unittest

from model.dataset import get_spacy_nlp


class SpacyTest(unittest.TestCase):
    def test_head(self):
        phrase = "vehicle to the right of woman"#the quick brown fox jumps over the lazy dog"
        nlp = get_spacy_nlp()

        doc = nlp(phrase)

        heads = [chunk.root.text for chunk in doc.noun_chunks]

        self.assertListEqual(heads, ["fox", "dog"])
