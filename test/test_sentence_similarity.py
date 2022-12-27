import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from utils.utils import load_vocabulary

s1 = "That is a happy person"
s2 = "That is a happy dog"  # 0.695
s3 = "That is a very happy person"  # 0.943
s4 = "Today is a sunny day"  # 0.257


class SentenceSimilarityTest(unittest.TestCase):
    def test__sentence_similarity__word_embedding_encoded_sentence(self):
        word_embedding = load_vocabulary(f"data/glove/glove.6B.50d.txt")
        indexer = word_embedding.word_indexer

        wv = nn.Embedding.from_pretrained(
            torch.from_numpy(word_embedding.vectors), freeze=True
        )

        get_tokens = lambda q: [max(indexer.index_of(word), 1) for word in q.split()]

        get_sentence_embedding = lambda s: wv(torch.tensor(get_tokens(s))).mean(dim=0)

        s1_emb = get_sentence_embedding(s1)
        s2_emb = get_sentence_embedding(s2)
        s3_emb = get_sentence_embedding(s3)
        s4_emb = get_sentence_embedding(s4)

        sim_s1_s2 = torch.cosine_similarity(s1_emb, s2_emb, dim=0)
        sim_s1_s3 = torch.cosine_similarity(s1_emb, s3_emb, dim=0)
        sim_s1_s4 = torch.cosine_similarity(s1_emb, s4_emb, dim=0)

        self.assertAlmostEqual(sim_s1_s2.item(), 0.9673, places=4)
        self.assertAlmostEqual(sim_s1_s3.item(), 0.9934, places=4)
        self.assertAlmostEqual(sim_s1_s4.item(), 0.9071, places=4)

    def test__sentence_similarity__minilm_encoded_sentence(self):
        def mean_pooling(model_output, attention_mask):
            # take attention mask into account for correct averaging
            token_embeddings = model_output[
                0
            ]  # first element of model_output contains all token embeddings
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        sentences = [s1, s2, s3, s4]

        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            padding=True,
            truncation=True,
        )

        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        encoded_input = tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=12,
            return_tensors="pt",
        )

        self.assertEqual(encoded_input["input_ids"].shape, (4, 12))

        with torch.no_grad():
            input_ids = encoded_input["input_ids"]
            attention_mask = encoded_input["attention_mask"]

            model_output = model(input_ids, attention_mask=attention_mask)

        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        self.assertEqual(sentence_embeddings.shape, (4, 384))

        s1_emb = sentence_embeddings[0]
        s2_emb = sentence_embeddings[1]
        s3_emb = sentence_embeddings[2]
        s4_emb = sentence_embeddings[3]

        sim_s1_s2 = torch.cosine_similarity(s1_emb, s2_emb, dim=0)
        sim_s1_s3 = torch.cosine_similarity(s1_emb, s3_emb, dim=0)
        sim_s1_s4 = torch.cosine_similarity(s1_emb, s4_emb, dim=0)

        self.assertAlmostEqual(sim_s1_s2.item(), 0.6945, places=3)
        self.assertAlmostEqual(sim_s1_s3.item(), 0.9428, places=3)
        self.assertAlmostEqual(sim_s1_s4.item(), 0.2571, places=3)
