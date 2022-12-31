import unittest

import torch
import torch.nn as nn

from model.loss import forward_all, forward_ce, forward_luca_random


class LossTest(unittest.TestCase):
    def test__loss__query_similarity_ones__same_result(self):
        predictions = torch.tensor([[0.1, 0.1], [0.3, 0.7]])  # [b, b]
        target = torch.tensor([0, 1])  # [b]

        batch_size = predictions.shape[0]

        def get_loss_1(predictions, target):
            pos_index = target.unsqueeze(-1)  # [b, 1]
            neg_index = (target + 1) % batch_size  # [b]
            neg_index = neg_index.unsqueeze(-1)  # [b, 1]
            pos_pred = torch.gather(predictions, 1, pos_index).squeeze(-1)
            neg_pred = torch.gather(predictions, 1, neg_index).squeeze(-1)
            loss = -torch.mean(pos_pred) + torch.mean(neg_pred)
            return loss

        def get_loss_2(predictions, target, query_weight):
            predictions_weighted = predictions * query_weight

            pos_index = target.unsqueeze(-1)  # [b, 1]
            pos_pred = torch.gather(predictions, 1, pos_index).squeeze(-1)  # [b]

            neg_index = (target + 1) % batch_size  # [b]
            neg_index = neg_index.unsqueeze(-1)  # [b, 1]
            neg_weight = torch.gather(query_weight, 1, neg_index).squeeze(-1)  # [b]
            neg_pred = torch.gather(predictions_weighted, 1, neg_index).squeeze(
                -1
            )  # [b]

            neg = torch.sum(neg_pred) / (torch.sum(neg_weight) + 1e-08)  # weighted mean

            loss = -torch.mean(pos_pred) + neg

            return loss

        query_weight = torch.ones_like(predictions)  # [b, b]

        loss_1 = get_loss_1(predictions, target)
        loss_2 = get_loss_2(predictions, target, query_weight)
        loss_3 = forward_luca_random(predictions, target, query_weight)

        self.assertAlmostEqual(loss_1.item(), loss_2.item())
        self.assertAlmostEqual(loss_1.item(), loss_3.item())

    def test__loss_all__query_similarity_ones__same_result(self):
        predictions = torch.tensor([[0.1, 0.1], [0.3, 0.7]])
        target = torch.tensor([0, 1])  # [b]

        batch_size = predictions.shape[0]

        def get_loss_1(predictions, target):
            pos_index = target.unsqueeze(-1)  # [b, 1]
            pos_pred = torch.gather(predictions, 1, pos_index).squeeze(-1)  # [b]
            neg_pred = (torch.sum(predictions, dim=-1) - pos_pred) / (
                batch_size - 1
            )  # [b]
            loss = -torch.mean(pos_pred) + torch.mean(neg_pred)
            return loss

        def get_loss_2(predictions, target, query_weight):
            query_weight_mask = 1 - torch.eye(batch_size)  # [b, b]
            query_weight = (
                query_weight * query_weight_mask
            )  # [b, b], remove pos predictions

            neg_pred_weighted = predictions * query_weight * query_weight_mask

            pos_index = target.unsqueeze(-1)  # [b, 1]
            pos_pred = torch.gather(predictions, 1, pos_index).squeeze(-1)  # [b]
            neg_pred = (torch.sum(neg_pred_weighted, dim=-1)) / torch.sum(
                query_weight, dim=-1
            )  # [b]
            loss = -torch.mean(pos_pred) + torch.mean(neg_pred)
            return loss

        query_weight = torch.ones_like(predictions)  # [b, b]

        loss_1 = get_loss_1(predictions, target)
        loss_2 = get_loss_2(predictions, target, query_weight)
        loss_3 = forward_all(predictions, target, query_weight)

        self.assertAlmostEqual(loss_1.item(), loss_2.item())
        self.assertAlmostEqual(loss_1.item(), loss_3.item())

    def test__loss_ce__query_similarity_ones__same_result(self):
        predictions = torch.tensor([[0.1, 0.1], [0.3, 0.7]])
        target = torch.tensor([0, 1])

        batch_size = predictions.shape[0]

        device = predictions.device

        def get_loss_1(predictions, target):
            ce = nn.CrossEntropyLoss(reduction="mean")
            loss = ce(predictions, target)
            return loss

        def get_loss_2(predictions, target, query_weight):
            ce = nn.CrossEntropyLoss(reduction="none")

            loss = ce(predictions, target)  # [b]

            neg_weight = query_weight + torch.eye(batch_size, device=device)  # [b, b]
            neg_weight = torch.sum(neg_weight, dim=-1)  # [b]

            loss = torch.sum(loss * neg_weight) / torch.sum(neg_weight)

            return loss

        query_weight = torch.ones_like(predictions)  # [b, b]

        loss_1 = get_loss_1(predictions, target)
        loss_2 = get_loss_2(predictions, target, query_weight)
        loss_3 = forward_ce(predictions, target, query_weight, cross_entropy_loss=nn.CrossEntropyLoss(reduction="none"))

        self.assertAlmostEqual(loss_1.item(), loss_2.item())
        self.assertAlmostEqual(loss_1.item(), loss_3.item())
