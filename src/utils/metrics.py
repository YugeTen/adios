from typing import Dict, List, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.metrics import adjusted_rand_score, f1_score, average_precision_score

def accuracy_at_k(
    outputs: torch.Tensor, targets: torch.Tensor, top_k: Sequence[int] = (1, 5)
) -> Dict:
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        outputs (torch.Tensor): output of a classifier (logits or probabilities).
        targets (torch.Tensor): ground truth labels.
        top_k (Sequence[int], optional): sequence of top k values to compute the accuracy over.
            Defaults to (1, 5).

    Returns:
        Dict:  accuracies at the desired k.
    """

    with torch.no_grad():
        maxk = max(top_k)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

    results = {}
    for k, r in zip(top_k, res):
        results[f"acc{k}"] = r

    return results

def multiclass_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> Dict:
    with torch.no_grad():
        # targets = targets["exists"]
        total = targets.shape[0]*outputs.shape[-1]
        outputs = torch.sigmoid(outputs)
        predicted = torch.round(outputs)
        correct = (predicted == targets).sum()

        results = {"acc": correct/total}
        metrics = ['macro', 'micro', 'weighted']
        [results.update({
            f"f1_{metric}": torch.tensor(f1_score(predicted.cpu(), targets.cpu(), average=metric, zero_division=1.0)).to(targets.device)
        }) for metric in metrics]

    return results


def weighted_mean(outputs: List[Dict], key: str, batch_size_key: str) -> float:
    """Computes the mean of the values of a key weighted by the batch size.

    Args:
        outputs (List[Dict]): list of dicts containing the outputs of a validation step.
        key (str): key of the metric of interest.
        batch_size_key (str): key of batch size values.

    Returns:
        float: weighted mean of the values of a key
    """

    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
    return value.squeeze(0)


def average_ari(m_k, instances, foreground_only=False):
    """
    Computes the average adjusted random index.
    Adapted from: https://github.com/applied-ai-lab/genesis.
    """
    ari = []
    masks_stacked = torch.stack(m_k, dim=4).exp().detach()
    masks_split = torch.split(masks_stacked, 1, dim=0)
    # Loop over elements in batch
    for i, m in enumerate(masks_split):
        masks_pred = np.argmax(m.cpu().numpy(), axis=-1).flatten()
        masks_gt = instances[i].detach().cpu().numpy().flatten()
        if foreground_only:
            masks_pred = masks_pred[np.where(masks_gt > 0)]
            masks_gt = masks_gt[np.where(masks_gt > 0)]
        score = adjusted_rand_score(masks_pred, masks_gt)
        ari.append(score)
    return torch.tensor(sum(ari)/len(ari)), torch.tensor(ari)


def iou_binary(mask_A, mask_B, debug=False):
    """
    Computes the binary IOU score.
    Adapted from: https://github.com/applied-ai-lab/genesis.
    """
    if debug:
        assert mask_A.shape == mask_B.shape
        assert mask_A.dtype == torch.bool
        assert mask_B.dtype == torch.bool
    intersection = (mask_A * mask_B).sum((1, 2, 3))
    union = (mask_A + mask_B).sum((1, 2, 3))
    # Return -100 if union is zero, else return IOU
    return torch.where(union == 0, torch.tensor(-100.0),
                       intersection.float() / union.float())

def average_segcover(segA, segB, ignore_background=False):
    """
    Covering of segA by segB
    segA.shape = [batch size, 1, img_dim1, img_dim2]
    segB.shape = [batch size, 1, img_dim1, img_dim2]

    scale: If true, take weighted mean over IOU values proportional to the
           the number of pixels of the mask being covered.

    Assumes labels in segA and segB are non-negative integers.
    Negative labels will be ignored.
    Adapted from: https://github.com/applied-ai-lab/genesis.
    """

    assert segA.shape == segB.shape, f"{segA.shape} - {segB.shape}"
    assert segA.shape[1] == 1 and segB.shape[1] == 1
    bsz = segA.shape[0]
    nonignore = (segA >= 0)

    mean_ious, mean_aps = torch.tensor(bsz*[0.0]), torch.tensor(bsz*[0.0])
    N, B = torch.tensor(bsz*[0]), segA.shape[0]
    scaled_ious, scaled_aps = torch.tensor(bsz*[0.0]), torch.tensor(bsz*[0.0])
    scaling_sum = torch.tensor(bsz*[0])

    # Find unique label indices to iterate over
    if ignore_background:
        iter_segA = torch.unique(segA[segA > 0]).tolist()
    else:
        iter_segA = torch.unique(segA[segA >= 0]).tolist()
    iter_segB = torch.unique(segB[segB >= 0]).tolist()
    # Loop over segA
    for i in iter_segA:
        binaryA = segA == i
        if not binaryA.any():
            continue
        max_iou, max_ap = torch.tensor(bsz*[0.0]), torch.tensor(bsz*[0.0]).float()
        # Loop over segB to find max IOU
        for j in iter_segB:
            # Do not penalise pixels that are in ignore regions
            binaryB = (segB == j) * nonignore
            if not binaryB.any():
                continue
            iou = iou_binary(binaryA, binaryB)
            ap = []
            for bA, bB in zip(binaryA, binaryB):
                bA, bB = bA.view(-1), bB.view(-1)
                if bA.sum() == 0:
                    ap_ = -1 if bB.sum() != 0 else 1
                else:
                    ap_ = average_precision_score(bA, bB)
                ap.append(ap_)
            ap = torch.tensor(ap).float()
            # ap = average_precision_score(binaryA.view(B, -1), binaryB.view(B, -1), average=None)
            # ap = torch.tensor(ap).mean(-1).float()

            max_iou = torch.where(iou > max_iou, iou, max_iou)
            max_ap = torch.where(ap > max_ap, ap, max_ap)
        # Accumulate scores
        mean_ious += max_iou
        scaled_ious += binaryA.sum((1, 2, 3)).float() * max_iou
        mean_aps += max_ap
        scaled_aps += binaryA.sum((1, 2, 3)).float() * max_ap

        N = torch.where(binaryA.sum((1, 2, 3)) > 0, N+1, N)
        scaling_sum += binaryA.sum((1, 2, 3))

    # Compute coverage
    mean_sc = mean_ious / torch.max(N, torch.tensor(1)).float()
    scaled_sc = scaled_ious / torch.max(scaling_sum, torch.tensor(1)).float()
    mean_ap = mean_aps / torch.max(N, torch.tensor(1)).float()
    scaled_ap = scaled_aps / torch.max(scaling_sum, torch.tensor(1)).float()

    # Sanity check
    assert (mean_sc >= 0).all() and (mean_sc <= 1).all(), mean_sc
    assert (scaled_sc >= 0).all() and (scaled_sc <= 1).all(), scaled_sc
    assert (mean_ious[N == 0] == 0).all()
    assert (mean_ious[nonignore.sum((1, 2, 3)) == 0] == 0).all()
    assert (scaled_ious[N == 0] == 0).all()
    assert (scaled_ious[nonignore.sum((1, 2, 3)) == 0] == 0).all()

    # Return mean over batch dimension
    return mean_sc.mean(0), scaled_sc.mean(0), mean_ap.mean(0), scaled_ap.mean(0)


class Entropy(nn.Module):
    """
    Computes the entropy.
    """
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        return b.sum()