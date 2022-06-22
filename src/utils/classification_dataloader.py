import os
from typing import Callable, Optional, Tuple

from torch import nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder

from src.utils.clevr import ClevrDataset

def prepare_transforms(dataset: str, size: int) -> Tuple[nn.Module, nn.Module]:
    """Prepares pre-defined train and test transformation pipelines for some datasets.

    Args:
        dataset (str): dataset name.

    Returns:
        Tuple[nn.Module, nn.Module]: training and validation transformation pipelines.
    """

    cifar_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomCrop(size, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
    }

    stl_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    imagenet_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(size),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        ),
    }

    clevr_pipeline = {
        "T_train": transforms.Compose([
                # crop_tf,
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.float()),
            ]),
        "T_val": transforms.Compose([
                # crop_tf,
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.float()),
            ])
    }

    pipelines = {
        "cifar10": cifar_pipeline,
        "cifar100": cifar_pipeline,
        "stl10": stl_pipeline,
        "imagenet100s": imagenet_pipeline,
        "imagenet100": imagenet_pipeline,
        "imagenet": imagenet_pipeline,
        "clevr": clevr_pipeline,
    }

    assert dataset in pipelines

    pipeline = pipelines[dataset]
    T_train = pipeline["T_train"]
    T_val = pipeline["T_val"]

    return T_train, T_val


def prepare_datasets(
    dataset: str,
    T_train: Callable,
    T_val: Callable,
    load_masks: bool,
    data_dir: Optional[str] = None,
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
    morphology: Optional[str] = "none",
) -> Tuple[Dataset, Dataset]:
    """Prepares train and val datasets.

    Args:
        dataset (str): dataset name.
        T_train (Callable): pipeline of transformations for training dataset.
        T_val (Callable): pipeline of transformations for validation dataset.
        data_dir Optional[str]: path where to download/locate the dataset.
        train_dir Optional[str]: subpath where the training data is located.
        val_dir Optional[str]: subpath where the validation data is located.

    Returns:
        Tuple[Dataset, Dataset]: training dataset and validation dataset.
    """

    if data_dir is None:
        sandbox_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        data_dir = os.path.join(sandbox_dir, "datasets")

    dataset_name = "imagenet100" if dataset == "imagenet100s" else dataset
    if train_dir is None:
        train_dir = f"{dataset_name}/train"
    if val_dir is None:
        val_dir = f"{dataset_name}/val"


    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = DatasetClass(
            os.path.join(data_dir, train_dir),
            train=True,
            download=True,
            transform=T_train,
        )

        val_dataset = DatasetClass(
            os.path.join(data_dir, val_dir),
            train=False,
            download=True,
            transform=T_val,
        )

    elif dataset == "stl10":
        train_dataset = STL10(
            os.path.join(data_dir, train_dir),
            split="train",
            download=True,
            transform=T_train,
        )
        val_dataset = STL10(
            os.path.join(data_dir, val_dir),
            split="test",
            download=True,
            transform=T_val,
        )

    elif dataset in ["imagenet", "imagenet100", "imagenet100s"]:
        train_dataset = ImageFolder(os.path.join(data_dir, train_dir), T_train)
        val_dataset = ImageFolder(os.path.join(data_dir, val_dir), T_val)

    elif dataset == "clevr":
        train_dataset = ClevrDataset(
            data_dir, T_train,
            split='train',
            morph=morphology,
            load_masks=load_masks
        )
        val_dataset = ClevrDataset(
            data_dir, T_val,
            split='val',
            morph=morphology,
            load_masks=load_masks
        )
    return train_dataset, val_dataset


def prepare_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Wraps a train and a validation dataset with a DataLoader.

    Args:
        train_dataset (Dataset): object containing training data.
        val_dataset (Dataset): object containing validation data.
        batch_size (int): batch size.
        num_workers (int): number of parallel workers.
    Returns:
        Tuple[DataLoader, DataLoader]: training dataloader and validation dataloader.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader



def prepare_data(
    dataset: str,
    size: int,
    load_masks: Optional[bool] = False,
    data_dir: Optional[str] = None,
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    morphology: Optional[str] = "none",
) -> Tuple[DataLoader, DataLoader]:
    """Prepares transformations, creates dataset objects and wraps them in dataloaders.

    Args:
        dataset (str): dataset name.
        data_dir (Optional[str], optional): path where to download/locate the dataset.
            Defaults to None.
        train_dir (Optional[str], optional): subpath where the training data is located.
            Defaults to None.
        val_dir (Optional[str], optional): subpath where the validation data is located.
            Defaults to None.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of parallel workers. Defaults to 4.

    Returns:
        Tuple[DataLoader, DataLoader]: prepared training and validation dataloader;.
    """

    T_train, T_val = prepare_transforms(dataset, size)
    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T_train,
        T_val,
        load_masks,
        data_dir=data_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        morphology=morphology,
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader
