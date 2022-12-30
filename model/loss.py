import torch
import torch.nn as nn


class WeakVtgLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        """
		:param args: args from command line
		"""
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

        self.loss_strategy = args.loss_strategy
        self.do_negative_weighting = args.do_negative_weighting
        self.loss_sigmoid_slope = args.loss_sigmoid_slope

    def forward(self, predictions, target, query_similarity):
        """
        :param predictions: [b, b]
        :param target: [b]
        :param query_similarity: [b, b]
        """
        query_weight = -1 * query_similarity
        query_weight = torch.sigmoid(query_weight * self.loss_sigmoid_slope)

        prediction_query_weight = query_weight if self.do_negative_weighting else None

        if self.loss_strategy == "luca":
            loss = forward_luca(
                predictions, target, query_weight=prediction_query_weight
            )
        elif self.loss_strategy == "all":
            loss = forward_all(
                predictions, target, query_weight=prediction_query_weight
            )
        elif self.loss_strategy == "ce":
            loss = forward_ce(
                predictions,
                target,
                query_weight=prediction_query_weight,
                cross_entropy_loss=self.ce_loss,
            )
        else:
            raise ValueError(f"Invalid loss strategy '{self.loss_strategy}'.")

        return loss


def forward_luca(predictions, target, query_weight=None):
    """
    :param predictions: [b, b]
    :param target: [b]
    :param query_weight: [b, b]
    """
    if query_weight is None:
        query_weight = torch.ones_like(predictions)

    batch_size = predictions.shape[0]

    predictions_weighted = predictions * query_weight

    pos_index = target.unsqueeze(-1)  # [b, 1]
    pos_pred = torch.gather(predictions, 1, pos_index).squeeze(-1)  # [b]

    neg_index = (target + 1) % batch_size  # [b]
    neg_index = neg_index.unsqueeze(-1)  # [b, 1]
    neg_weight = torch.gather(query_weight, 1, neg_index).squeeze(-1)  # [b]
    neg_pred = torch.gather(predictions_weighted, 1, neg_index).squeeze(-1)  # [b]

    neg = torch.sum(neg_pred) / (torch.sum(neg_weight) + 1e-08)  # weighted mean

    loss = -torch.mean(pos_pred) + neg

    return loss


def forward_all(predictions, target, query_weight=None):
    """
    :param predictions: [b, b]
    :param target: [b]
    :param query_weight: query weight used to multiply predictions [b, b]
    :param query_similarity: query similarity eventually used to retrieve negative examples [b, b]
    """
    if query_weight is None:
        query_weight = torch.ones_like(predictions)

    batch_size = predictions.shape[0]

    # given prediction a tensor of [b, b], for each example (row) in batch we
    # gather negative examples as all the other example in row except the positive
    # one, i.e. the one in the diagonal

    neg_query_weight_mask = 1 - torch.eye(batch_size)  # [b, b]
    neg_query_weight = query_weight * neg_query_weight_mask  # [b, b]

    neg_pred_weighted = predictions * neg_query_weight

    pos_index = target.unsqueeze(-1)  # [b, 1]
    pos_pred = torch.gather(predictions, 1, pos_index).squeeze(-1)  # [b]
    neg_pred = (torch.sum(neg_pred_weighted, dim=-1)) / torch.sum(
        neg_query_weight, dim=-1
    )  # [b]
    loss = -torch.mean(pos_pred) + torch.mean(neg_pred)
    return loss


def forward_ce(predictions, target, query_weight, cross_entropy_loss, *_, **__):
    if query_weight is None:
        query_weight = torch.ones_like(predictions)

    device = predictions.device
    batch_size = predictions.shape[0]

    loss = cross_entropy_loss(predictions, target)  # [b]

    neg_weight = query_weight + torch.eye(batch_size, device=device)  # [b, b]
    neg_weight = torch.sum(neg_weight, dim=-1)  # [b]

    loss = torch.sum(loss * neg_weight) / torch.sum(neg_weight)

    return loss
