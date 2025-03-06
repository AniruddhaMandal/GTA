import torch
import torch.nn as nn
import torch.nn.functional as F

def get_loss_fn(cfg):
    if(cfg.Train.loss_fn == 'cross-entropy'):
        return nn.CrossEntropyLoss()
    if(cfg.Train.loss_fn == 'multilable-cross-entropy'):
        return nn.BCEWithLogitsLoss()
    if(cfg.Train.loss_fn == "l1"):
        return nn.L1Loss()
    if(cfg.Train.loss_fn == "weighted-cross-entropy"):
        return weighted_cross_entropy

def weighted_cross_entropy(pred, true):
    """ Weighted cross-entropy for unbalanced classes.
        Code snippet taken from LRGB repo.  
    """
    # calculating label weights for weighted loss computation
    V = true.size(0)
    n_classes = pred.shape[1] if pred.ndim > 1 else 2
    label_count = torch.bincount(true)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()
    # multiclass
    if pred.ndim > 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true, weight=weight)
    # binary
    else:
        loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                    weight=weight[true])
        return loss