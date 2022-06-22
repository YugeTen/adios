import json
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.args.setup import parse_args_knn
from src.methods import METHODS
from src.utils.classification_dataloader import (
    prepare_dataloaders,
    prepare_datasets,
    prepare_transforms,
)
from src.utils.knn import WeightedKNNClassifier
from src.args.utils import IMG_SIZE_DATASET

@torch.no_grad()
def extract_features(loader: DataLoader, model: nn.Module) -> Tuple[torch.Tensor]:
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
def run_knn(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    k: int,
    T: float,
    distance_fx: str,
) -> Tuple[float]:
    """Runs offline knn on a train and a test dataset.
    Args:
        train_features (torch.Tensor, optional): train features.
        train_targets (torch.Tensor, optional): train targets.
        test_features (torch.Tensor, optional): test features.
        test_targets (torch.Tensor, optional): test targets.
        k (int): number of neighbors.
        T (float): temperature for the exponential. Only used with cosine
            distance.
        distance_fx (str): distance function.
    Returns:
        Tuple[float]: tuple containing the the knn acc@1 and acc@5 for the model.
    """

    # build knn
    knn = WeightedKNNClassifier(
        k=k,
        T=T,
        distance_fx=distance_fx,
    )
    # add features
    knn(
        train_features=train_features,
        train_targets=train_targets,
        test_features=test_features,
        test_targets=test_targets,
    )
    # compute
    acc1, acc5 = knn.compute()
    # free up memory
    del knn
    return acc1, acc5


def main():
    args = parse_args_knn()

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

    optim_acc1, optim_acc5, optim_params= 0., 0., []
    # run k-nn for all possible combinations of parameters
    for feat_type in args.feature_type:
        print(f"\n### {feat_type.upper()} ###")
        for k in args.k:
            for distance_fx in args.distance_function:
                temperatures = args.temperature if distance_fx == "cosine" else [None]
                for T in temperatures:
                    acc1, acc5 = run_knn(
                        train_features=train_features[feat_type],
                        train_targets=train_targets,
                        test_features=test_features[feat_type],
                        test_targets=test_targets,
                        k=k,
                        T=T,
                        distance_fx=distance_fx,
                    )
                    if acc1 > optim_acc1:
                        optim_acc1, optim_acc5 = acc1, acc5
                        optim_params = [k, distance_fx, T, distance_fx, feat_type]
    print(f"k-NN with params: distance_function={optim_params[3]}, feature_type={optim_params[4]}, "
          f"distance_fx={optim_params[1]}, k={optim_params[0]}, T={optim_params[2]} "
          f"has best acc1: {optim_acc1:6.3f}, acc5: {optim_acc5:6.3f}")

if __name__ == "__main__":
    main()
