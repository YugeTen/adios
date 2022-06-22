import os, json
import wandb
import argparse
from pathlib import Path
from argparse import Namespace

import pytorch_lightning as pl
from src.args.dataset import augmentations_args, dataset_args
from src.args.utils import additional_setup_linear, additional_setup_pretrain
from src.methods import METHODS
from src.utils.auto_resumer import AutoResumer
from src.utils.checkpointer import Checkpointer
from src.utils.blocks import str2bool
from src.utils.auto_mask import AutoMASK
try:
    from src.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

NEED_AUTO_MASK = ["simclr_adios", "simclr_adios_s", "simsiam_adios", "simsiam_adios_s",
                  'byol_adios', "byol_adios_s", ]
NEED_LOAD_MASK = ["simclr_gt", "simclr_rand_mask", *NEED_AUTO_MASK]

def parse_args_pretrain() -> argparse.Namespace:
    """Parses dataset, augmentation, pytorch lightning, model specific and additional args.

    First adds shared args such as dataset, augmentation and pytorch lightning args, then pulls the
    model name from the command and proceeds to add model specific args from the desired class. If
    wandb is enabled, it adds checkpointer args. Finally, adds additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    # add shared arguments
    dataset_args(parser)
    augmentations_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # add method-specific arguments
    parser.add_argument("--method", type=str)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # add model specific args
    parser = METHODS[temp_args.method].add_model_specific_args(parser)

    # add auto umap, auto mask args
    parser.add_argument("--auto_umap", type=str2bool, nargs='?',
                            const=True, default=False)
    parser.add_argument("--auto_mask", type=str2bool, nargs='?',
                            const=True, default=True)
    parser.add_argument("--load_masks", type=str2bool, nargs='?',
                            const=True, default=True)
    parser.add_argument("--auto_resume", type=str2bool, nargs='?',
                            const=False, default=False)
    parser.add_argument("--wandb_dir", type=str)
    parser.add_argument("--validation_frequency", type=int, default=1)
    parser.add_argument("--pretrained_dir", type=str, default=None)
    # optionally add checkpointer and AutoUMAP args
    temp_args, _ = parser.parse_known_args()
    if temp_args.wandb:
        parser = Checkpointer.add_checkpointer_args(parser)

    if _umap_available and temp_args.auto_umap:
        parser = AutoUMAP.add_auto_umap_args(parser)
    if temp_args.auto_mask:
        parser = AutoMASK.add_auto_mask_args(parser)
    if temp_args.auto_resume:
        parser = AutoResumer.add_autoresumer_args(parser)

    # parse args
    args = parser.parse_args()
    if temp_args.method not in NEED_AUTO_MASK:
        args.auto_mask = False
    if temp_args.method not in NEED_LOAD_MASK:
        args.load_masks = False
    if not hasattr(temp_args, "morph"):
        args.morph = 'none'

    # load pretrained model
    if args.pretrained_dir is not None:
        assert os.path.exists(args.pretrained_dir), \
            f"Pretrained model folder {args.pretrained_dir} does not exist!"
        args_path = os.path.join(args.pretrained_dir, "args.json")
        ckpt_paths = [os.path.join(args.pretrained_dir, ckpt)
                     for ckpt in os.listdir(Path(args.pretrained_dir))
                     if ckpt.endswith(".ckpt")]
        assert os.path.exists(args_path) and len(ckpt_paths)>0, \
            f"Pretrained model folder {args.pretrained_dir} is incomplete! " \
            f"Make sure there is a checkpoint file and args file in the directory."
        args.ckpt_path, args.args_path = ckpt_paths[0], args_path
        # load arguments
        with open(args_path) as f:
            pretrained_args = json.load(f)
        # some args needs to be loaded from pretrained models
        args = inherit_args_from_pretrained(args.__dict__, pretrained_args)

    # prepare arguments with additional setup
    additional_setup_pretrain(args)

    wandb_args = {
        "config": args,
        "project": args.project,
        "entity": args.entity,
        "dir": args.wandb_dir
    }
    if args.name != "none":
        wandb_args["name"] = args.name

    wandb.init(**wandb_args)
    c = wandb.config
    return Namespace(**c)

def inherit_args_from_pretrained(args, pretrained_args):
    """
    Update some args that determines model architectures to the ones used in pretrained models
    to avoid errors when loading the model.
    """
    assert pretrained_args["method"] == args["method"], \
    f"Pretrained model is {pretrained_args['method']} while current method is set as {args['method']}! " \
    f"Please choose the same method as the one used for pretrained model."

    common_args = ["N", "encoder", "mask_fbase", "unet_norm"]
    all_to_be_update_args = {
        "byol_adios": [*common_args, "output_dim", "pred_hidden_dim", "proj_hidden_dim"],
        "simsiam_adios": [*common_args, "output_dim", "pred_hidden_dim", "proj_hidden_dim"],
        "simclr_adios": [*common_args, "output_dim", "proj_hidden_dim"]
    }
    to_be_update_args = [val for key, val in all_to_be_update_args.items() if key in pretrained_args["method"]]

    for arg in to_be_update_args[0]:
        args.update({arg: pretrained_args[arg]})
    return Namespace(**args)

def parse_args_linear() -> argparse.Namespace:
    """Parses args for linear training, i.e. feature extractor.

    Returns:
        argparse.Namespace: a namespace containing all args needed for linear probing.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_feature_extractor", type=str)

    # add shared arguments
    dataset_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # linear model
    parser = METHODS["linear"].add_model_specific_args(parser)

    # THIS LINE IS KEY TO PULL WANDB
    temp_args, _ = parser.parse_known_args()

    # add checkpointer args (only if logging is enabled)
    if temp_args.wandb:
        parser = Checkpointer.add_checkpointer_args(parser)

    # parse args
    args = parser.parse_args()
    additional_setup_linear(args)

    return args

def parse_args_finetune() -> argparse.Namespace:
    """Parses args for finetuning, including feature extractor, and validation frequency.

    Returns:
        argparse.Namespace: a namespace containing all args needed for finetuning.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_feature_extractor", type=str)
    parser.add_argument("--validation_frequency", type=int, default=1)

    # add shared arguments
    dataset_args(parser)

    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # linear model
    parser = METHODS["supervised"].add_model_specific_args(parser)

    # THIS LINE IS KEY TO PULL WANDB
    temp_args, _ = parser.parse_known_args()

    # add checkpointer args (only if logging is enabled)
    if temp_args.wandb:
        parser = Checkpointer.add_checkpointer_args(parser)

    # parse args
    args = parser.parse_args()
    additional_setup_linear(args)

    return args


def parse_args_knn() -> argparse.Namespace:
    """Parses args for knn.

    Returns:
        argparse.Namespace: a namespace containing all args needed for knn.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_feature_extractor", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    parser.add_argument("--temperature", type=float, nargs="+",default=[0.5, 0.1, 0.05, 0.01, 0.005, 0.001])
    parser.add_argument("--distance_function", type=str, nargs="+",default=['cosine', 'euclidean'])
    parser.add_argument("--feature_type", type=str, nargs="+",default=['backbone', 'projector'])

    # add shared arguments
    dataset_args(parser)

    # parse args
    args = parser.parse_args()

    return args


def parse_args_clustering() -> argparse.Namespace:
    """Parses args for clustering.

    Returns:
        argparse.Namespace: a namespace containing all args needed for knn.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_feature_extractor", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--feature_type", type=str, nargs="+",default=['backbone', 'projector'])
    # add shared arguments
    dataset_args(parser)
    # parse args
    args = parser.parse_args()
    return args
