import torch
import torch.nn as nn
import torch.nn.functional as F


class WeakVtgLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        """
		:param args: args from command line
		"""
        # params
        self.CE_loss = nn.CrossEntropyLoss(reduction = "mean")
        self.loss_strategy = args.loss_strategy

    def forward(self, predictions, target, query_similarity):
        """
        :param predictions: [b, b]
        :param target: [b]
        """
        batch_size = predictions.shape[0]

        if self.loss_strategy == 'luca':
            pos_index = target.unsqueeze(-1)     # [b, 1]
            neg_index = (target + 1) % batch_size  # [b]
            neg_index = neg_index.unsqueeze(-1)     # [b, 1]
            pos_pred = torch.gather(predictions, 1, pos_index).squeeze(-1)
            neg_pred = torch.gather(predictions, 1, neg_index).squeeze(-1)
            
            w = torch.gather(query_similarity, dim=-1,index= neg_index)  # [b, 1]
            w = torch.gather(w, dim=0, index=pos_index).squeeze(-1)      # [b]

            neg = torch.sum(neg_pred * w) / torch.sum(w)
            
            loss = - torch.mean(pos_pred) + neg
        elif self.loss_strategy == 'all':
            predictions_weighted = predictions * query_similarity
            pos_index = target.unsqueeze(-1)     # [b, 1]
            pos_pred = torch.gather(predictions, 1, pos_index).squeeze(-1)  # [b]
            neg_pred = (torch.sum(predictions_weighted, dim=-1) - pos_pred) / (batch_size - 1)   # [b]
            loss = - torch.mean(pos_pred)  + torch.mean(neg_pred)
        elif self.loss_strategy == 'ce':
            loss = self.CE_loss(predictions, target)
        else:
            print("Error, loss_strategy '{}' not defined. ".format(self.loss_strategy))
            exit(1)
        return loss
        
