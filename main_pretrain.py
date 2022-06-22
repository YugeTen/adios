import os, sys
import numpy as np
from pprint import pprint

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from src.args.setup import parse_args_pretrain
from src.methods import METHODS
from src.utils.auto_resumer import AutoResumer

try:
    from src.methods.dali import PretrainABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

from src.utils.auto_mask import AutoMASK
try:
    from src.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

from src.utils.checkpointer import Checkpointer
from src.utils.classification_dataloader import prepare_data as prepare_data_classification
from src.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_multicrop_transform,
    prepare_n_crop_transform,
    prepare_transform,
)


def main():
    seed = np.random.randint(0, 2**32)
    seed_everything(seed)
    args = parse_args_pretrain()
    if sys.gettrace() is not None:
        args.num_workers = 0

    assert args.method in METHODS, f"Choose from {METHODS.keys()}"

    MethodClass = METHODS[args.method]
    if args.dali:
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with [dali]."
        MethodClass = type(f"Dali{MethodClass.__name__}", (MethodClass, PretrainABC), {})
    model = MethodClass(**args.__dict__)

    # add img size to transform kwargs
    args.transform_kwargs.update({"size": args.img_size})

    # contrastive dataloader
    if not args.dali:
        # asymmetric augmentations
        if args.unique_augs > 1:
            transform = [
                prepare_transform(args.dataset, multicrop=args.multicrop, **kwargs)
                for kwargs in args.transform_kwargs
            ]
        else:
            transform = prepare_transform(
                args.dataset, multicrop=args.multicrop, **args.transform_kwargs
            )

        if args.debug_augmentations:
            print("Transforms:")
            pprint(transform)

        if args.multicrop:
            assert not args.unique_augs == 1

            if args.dataset in ["cifar10", "cifar100"]:
                size_crops = [32, 24]
            elif args.dataset == "stl10":
                size_crops = [96, 58]
            # imagenet or custom dataset
            else:
                size_crops = [224, 96]

            transform = prepare_multicrop_transform(
                transform, size_crops=size_crops, n_crops=[args.n_crops, args.n_small_crops]
            )
        else:
            if args.n_crops != 2:
                assert args.method == "wmse"

            transform = prepare_n_crop_transform(transform, n_crops=args.n_crops)

        train_dataset = prepare_datasets(
            args.dataset,
            transform,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            morphology=args.morph,
            load_masks=args.load_masks,
        )
        train_loader = prepare_dataloader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )

    # normal dataloader for when it is available
    _, val_loader = prepare_data_classification(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        size=args.img_size,
        load_masks = args.load_masks,
    )

    callbacks = []

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name,
            project=args.project,
            entity=args.entity,
            offline=args.offline,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, args.method),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

        if args.auto_umap:
            assert (
                _umap_available
            ), "UMAP is not currently avaiable, please install it first with [umap]."
            auto_umap = AutoUMAP(
                args,
                logdir=os.path.join(args.auto_umap_dir, args.method),
                frequency=args.auto_umap_frequency,
            )
            callbacks.append(auto_umap)

        if args.auto_mask:
            auto_mask = AutoMASK(
                args,
                logdir=os.path.join(args.auto_mask_dir, args.method),
                frequency=args.auto_mask_frequency,
            )
            callbacks.append(auto_mask)

    if args.auto_resume and args.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(args.checkpoint_dir, args.method),
            load_dir=args.load_dir,
            search_in_checkpoint_dir=args.search_in_checkpoint_dir
        )
        resume_from_checkpoint = auto_resumer.find_checkpoint(args)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            args.resume_from_checkpoint = resume_from_checkpoint


    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=False),
        checkpoint_callback=False,
        terminate_on_nan=True,
        accelerator="ddp",
        check_val_every_n_epoch=args.validation_frequency,
    )

    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()

