import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.args.setup import parse_args_clustering
from src.methods import METHODS
from src.utils.classification_dataloader import (
    prepare_dataloaders,
    prepare_datasets,
    prepare_transforms,
)
from src.args.utils import IMG_SIZE_DATASET

from sklearn.metrics import fowlkes_mallows_score, \
    adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import k_means


@torch.no_grad()
def extract_features(loader: DataLoader, model: nn.Module):
    """Extract features from a data loader using a model.
    Args:
        loader (DataLoader): dataloader for a dataset.
        model (nn.Module): torch module used to extract features.
    Returns:
        Tuple(torch.Tensor): tuple containing the backbone features, projector features and labels.
    """
    model.eval()
    backbone_features, proj_features, labels = [], [], []
    for im, lab in tqdm(loader):
        im = im.cuda(non_blocking=True)
        lab = lab.cuda(non_blocking=True)
        outs = model(im)
        backbone_feature = outs["feats"].detach()
        proj_feature = outs["z"].detach()
        backbone_features.append(backbone_feature)
        proj_features.append(proj_feature)
        labels.append(lab)
    model.train()
    backbone_features = torch.cat(backbone_features)
    proj_features = torch.cat(proj_features)
    labels = torch.cat(labels)
    return backbone_features, proj_features, labels


@torch.no_grad()
def run_clustering(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
):
    """Runs offline knn on a train and a test dataset.
    Args:
        train_features (torch.Tensor, optional): train features.
        train_targets (torch.Tensor, optional): train targets.
        test_features (torch.Tensor, optional): test features.
        test_targets (torch.Tensor, optional): test targets.
    Returns:
        FMI, ARI, NMI score for train and test
    """
    
    n_class = train_targets.max().cpu()
    _, train_preds, _ = k_means(train_features.cpu(), n_clusters=n_class)
    _, test_preds, _ = k_means(test_features.cpu(), n_clusters=n_class)
    train_targets = train_targets.cpu()
    test_targets = test_targets.cpu()

    train_fmi = fowlkes_mallows_score(train_targets, train_preds)
    test_fmi = fowlkes_mallows_score(test_targets, test_preds)

    train_ari = adjusted_rand_score(train_targets, train_preds)
    test_ari = adjusted_rand_score(test_targets, test_preds)

    train_nmi = normalized_mutual_info_score(train_targets, train_preds)
    test_nmi = normalized_mutual_info_score(test_targets, test_preds)
    return train_fmi, test_fmi, train_ari, test_ari, train_nmi, test_nmi


def main():
    args = parse_args_clustering()
    print(f"Examining {args.pretrained_feature_extractor}")

    # build paths
    ckpt_dir = Path(args.pretrained_feature_extractor)
    args_path = ckpt_dir / "args.json"
    ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][0]

    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)

    # build the model
    model = METHODS[method_args["method"]].load_from_checkpoint(
        ckpt_path, strict=False, **method_args
    )
    model.cuda()

    # prepare data
    _, T = prepare_transforms(args.dataset, IMG_SIZE_DATASET[args.dataset])
    train_dataset, val_dataset = prepare_datasets(
        args.dataset,
        T_train=T,
        T_val=T,
        load_masks=False,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # extract train features
    train_features_bb, train_features_proj, train_targets = \
        extract_features(train_loader, model)
    train_features = {"backbone": train_features_bb, "projector": train_features_proj}

    # extract test features
    test_features_bb, test_features_proj, test_targets = extract_features(val_loader, model)
    test_features = {"backbone": test_features_bb, "projector": test_features_proj}

    # run k-nn for all possible combinations of parameters
    for feat_type in args.feature_type:
        print(f"\n### {feat_type.upper()} ###")

        train_fmi, test_fmi, train_ari, test_ari, train_nmi, test_nmi = run_clustering(
            train_features=train_features[feat_type],
            train_targets=train_targets,
            test_features=test_features[feat_type],
            test_targets=test_targets,
        )

        print(f"FMI => train:{train_fmi:6.4f}, test:{test_fmi:6.4f} \n"
              f"ARI => train:{train_ari:6.4f}, test:{test_ari:6.4f} \n"
              f"NMI => train:{train_nmi:6.4f}, test:{test_nmi:6.4f}")

if __name__ == "__main__":
    main()

