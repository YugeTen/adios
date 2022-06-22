from argparse import ArgumentParser
from src.utils.blocks import str2bool

def dataset_args(parser: ArgumentParser):
    """Adds dataset-related arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add dataset args to.
    """

    parser.add_argument("--dataset", type=str)

    # dataset path
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)

    # dali (imagenet-100/imagenet/custom only)
    parser.add_argument("--dali", type=str2bool, nargs='?',
                            const=True, default=False)
    parser.add_argument("--dali_device", type=str, default="gpu")

    # custom dataset only
    parser.add_argument("--no_labels", type=str2bool, nargs='?',
                            const=True, default=True)

def augmentations_args(parser: ArgumentParser):
    """Adds augmentation-related arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add augmentation args to.
    """

    # cropping
    parser.add_argument("--multicrop", type=str2bool, nargs='?',
                            const=True)
    parser.add_argument("--n_crops", type=int, default=2)
    parser.add_argument("--n_small_crops", type=int, default=0)

    # augmentations
    parser.add_argument("--brightness", type=float, nargs="+", default=[0.8])
    parser.add_argument("--contrast", type=float, nargs="+", default=[0.8])
    parser.add_argument("--saturation", type=float, nargs="+", default=[0.8])
    parser.add_argument("--hue", type=float, nargs="+", default=[0.2])
    parser.add_argument("--gaussian_prob", type=float, default=[0.5], nargs="+")
    parser.add_argument("--solarization_prob", type=float, default=[0.0], nargs="+")
    parser.add_argument("--min_scale", type=float, default=[0.08], nargs="+")

    # for imagenet or custom dataset
    parser.add_argument("--size", type=int, default=[224], nargs="+")

    # for custom dataset
    parser.add_argument("--mean", type=float, default=[0.485, 0.456, 0.406], nargs="+")
    parser.add_argument("--std", type=float, default=[0.228, 0.224, 0.225], nargs="+")

    # debug
    parser.add_argument("--debug_augmentations", type=str2bool, nargs='?',
                            const=True)
