from functools import partial

import torch
from torchmetrics import MeanAbsoluteError
from torcheval.metrics.functional import multiclass_f1_score
import numpy as np
from sklearn.metrics import average_precision_score

def get_metric_fn(cfg):
    if cfg.Train.metric == "accuracy":
        return batch_accuracy
    if cfg.Train.metric == "average-precision":
        return eval_ap
    if cfg.Train.metric == "mae":
        return MeanAbsoluteError().to(cfg.Device)
    if cfg.Train.metric == "mrr":
        return compute_mrr
    if cfg.Train.metric == "macro-multiclass-f1":
        return partial(multiclass_f1_score, num_classes=cfg.Data.output_dim, average="macro")
    
def eval_ap(y_pred,y_true):
    """ Code taken form LRGB repo. 
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    ap_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            ap_list.append(ap)
    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')
    return  sum(ap_list) / len(ap_list)

def batch_accuracy(X:torch.Tensor,y:torch.Tensor) -> float:
    return torch.sum(torch.argmax(X, dim=-1) == y)/len(y)

def compute_mrr(batch)-> dict:
    stats = {}
    for data in batch.to_data_list():
        # print(data.num_nodes)
        # print(data.edge_index_labeled)
        # print(data.edge_label)
        pred = data.x @ data.x.transpose(0, 1)
        # print(pred.shape)

        pos_edge_index = data.edge_label_index[:, data.edge_label == 1]
        num_pos_edges = pos_edge_index.shape[1]
        # print(pos_edge_index, num_pos_edges)

        pred_pos = pred[pos_edge_index[0], pos_edge_index[1]]
        # print(pred_pos)

        if num_pos_edges > 0:
            # raw MRR (original metric)
            neg_mask = torch.ones([num_pos_edges, data.num_nodes], dtype=torch.bool)
            neg_mask[torch.arange(num_pos_edges), pos_edge_index[1]] = False
            pred_neg = pred[pos_edge_index[0]][neg_mask].view(num_pos_edges, -1)
            mrr_list = _eval_mrr(pred_pos, pred_neg, 'torch', suffix='')
            # print(pred_neg, pred_neg.shape)

            # filtered MRR
            pred_masked = pred.clone()
            pred_masked[pos_edge_index[0], pos_edge_index[1]] -= float("inf")
            pred_neg = pred_masked[pos_edge_index[0]]
            mrr_list.update(_eval_mrr(pred_pos, pred_neg, 'torch', suffix='_filtered'))

            # extended filter without self-loops
            pred_masked.fill_diagonal_(-float("inf"))
            pred_neg = pred_masked[pos_edge_index[0]]
            mrr_list.update(_eval_mrr(pred_pos, pred_neg, 'torch', suffix='_filtered_noself'))
        else:
            # Return empty stats.
            mrr_list = _eval_mrr(pred_pos, pred_pos, 'torch')

        # print(mrr_list)
        for key, val in mrr_list.items():
            if key.endswith('_list'):
                key = key[:-len('_list')]
                val = float(val.mean().item())
            if np.isnan(val):
                val = 0.
            if key not in stats:
                stats[key] = [val]
            else:
                stats[key].append(val)
            # print(key, val)
        # print('-' * 80)

    # print('=' * 80, batch.split)
    batch_stats = {}
    for key, val in stats.items():
        mean_val = sum(val) / len(val)
        batch_stats[key] = mean_val
        # print(f"{key}: {mean_val}")
    return batch_stats

def _eval_mrr(y_pred_pos, y_pred_neg, type_info, suffix=''):
    """ Compute Hits@k and Mean Reciprocal Rank (MRR).

    Implementation from OGB:
    https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/evaluate.py

    Args:
        y_pred_neg: array with shape (batch size, num_entities_neg).
        y_pred_pos: array with shape (batch size, )
    """

    if type_info == 'torch':
        y_pred = torch.cat([y_pred_pos.view(-1, 1), y_pred_neg], dim=1)
        argsort = torch.argsort(y_pred, dim=1, descending=True)
        ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
        ranking_list = ranking_list[:, 1] + 1
        hits1_list = (ranking_list <= 1).to(torch.float)
        hits3_list = (ranking_list <= 3).to(torch.float)
        hits10_list = (ranking_list <= 10).to(torch.float)
        mrr_list = 1. / ranking_list.to(torch.float)

        return {f'hits@1{suffix}_list': hits1_list,
                f'hits@3{suffix}_list': hits3_list,
                f'hits@10{suffix}_list': hits10_list,
                f'mrr{suffix}_list': mrr_list}
    else:
        y_pred = np.concatenate([y_pred_pos.reshape(-1, 1), y_pred_neg],
                                axis=1)
        argsort = np.argsort(-y_pred, axis=1)
        ranking_list = (argsort == 0).nonzero()
        ranking_list = ranking_list[1] + 1
        hits1_list = (ranking_list <= 1).astype(np.float32)
        hits3_list = (ranking_list <= 3).astype(np.float32)
        hits10_list = (ranking_list <= 10).astype(np.float32)
        mrr_list = 1. / ranking_list.astype(np.float32)

        return {f'hits@1{suffix}_list': hits1_list,
                f'hits@3{suffix}_list': hits3_list,
                f'hits@10{suffix}_list': hits10_list,
                f'mrr{suffix}_list': mrr_list}