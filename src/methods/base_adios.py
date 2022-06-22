import torch
from functools import partial
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Callable, Sequence

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.lars import LARSWrapper
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from src.utils.metrics import average_ari, average_segcover, weighted_mean
from src.methods.base import BaseModel, BaseMomentumModel


def static_lr(
    get_lr: Callable, param_group_indexes: Sequence[int], lrs_to_replace: Sequence[float]
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class BaseADIOSModel(BaseModel):
    def __init__(self, dataset, train_mask_epoch=0, **kwargs):
        self.use_mask = (dataset in ["clevr", "coco"])
        super().__init__(**kwargs)
        # add summed, max metrics
        self.extra_metric_keys = []
        self.mask_metric_keys = ["mask_ari", "mask_ari_fg", "mask_msc","mask_msc_fg",
                                 "mask_ssc", "mask_ssc_fg", "mask_map", "mask_sap",
                                 "mask_map_fg", "mask_sap_fg"]
        self.train_mask_epoch = train_mask_epoch

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds basic momentum arguments that are shared for all methods.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parent_parser = super(BaseADIOSModel, BaseADIOSModel).add_model_specific_args(
            parent_parser
        )
        parser = parent_parser.add_argument_group("base")
        parser.add_argument("--train_mask_epoch", default=0, type=int)

        return parent_parser


    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        # select optimizer
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam)")

        # create optimizer
        if isinstance(self.learnable_params, dict):
            # collect learnable parameters
            idxs_no_scheduler = [
                i for i, m in enumerate(self.learnable_params['inpainter']) if m.pop("static_lr", False)
            ]

            optimizer = [optimizer(
                self.learnable_params['inpainter'],
                lr=self.lr,
                weight_decay=self.weight_decay,
                **self.extra_optimizer_args,
            ),
            optimizer(
                    self.learnable_params['mask'],
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    **self.extra_optimizer_args,
                )]
        else:
            idxs_no_scheduler = [
                i for i, m in enumerate(self.learnable_params) if m.pop("static_lr", False)
            ]
            optimizer = [optimizer(
                self.learnable_params,
                lr=self.lr,
                weight_decay=self.weight_decay,
                **self.extra_optimizer_args,
            )]

        # optionally wrap with lars
        if self.lars:
            optimizer = [LARSWrapper(
                opt,
                eta=self.eta_lars,
                clip=self.grad_clip_lars,
                exclude_bias_n_norm=self.exclude_bias_n_norm,
            ) for opt in optimizer]

        if self.scheduler == "none":
            return optimizer # todo: might need some touch up due to new optimiser structure
        else:
            if self.scheduler == "warmup_cosine":
                scheduler = [
                    LinearWarmupCosineAnnealingLR(
                        optimizer[0],
                        warmup_epochs=self.warmup_epochs,
                        max_epochs=self.max_epochs-self.train_mask_epoch,
                        warmup_start_lr=self.warmup_start_lr,
                        eta_min=self.min_lr,
                    ),
                    LinearWarmupCosineAnnealingLR(
                        optimizer[1],
                        warmup_epochs=self.warmup_epochs,
                        max_epochs=self.max_epochs,
                        warmup_start_lr=self.warmup_start_lr,
                        eta_min=self.min_lr,
                    )]
            elif self.scheduler == "cosine":
                scheduler = [CosineAnnealingLR(optimizer[0], self.max_epochs-self.train_mask_epoch, eta_min=self.min_lr),
                             CosineAnnealingLR(optimizer[1], self.max_epochs, eta_min=self.min_lr)]
            elif self.scheduler == "step":
                scheduler = [MultiStepLR(opt, self.lr_decay_steps) for opt in optimizer]
            else:
                raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

            if idxs_no_scheduler:
                get_lr = scheduler[0].get_lr if isinstance(scheduler, list) else scheduler.get_lr
                partial_fn = partial(
                    static_lr,
                    get_lr=get_lr,
                    param_group_indexes=idxs_no_scheduler,
                    lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
                )
                if isinstance(scheduler, list):
                    scheduler[0].get_lr = partial_fn
                else:
                    scheduler.get_lr = partial_fn

            return optimizer, scheduler


    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        # call parent class to get results from standard metrics
        base_metrics = super().validation_step(batch, batch_idx)

        seg_metrics = {}
        if self.use_mask:
            # forward pass, get masks
            X, targets = batch
            with torch.no_grad():
                mask_enc_feat = self.mask_encoder(X)
                masks = self.mask_head(mask_enc_feat)
                pred_masks = torch.chunk(masks, masks.shape[1], dim=1)
            gt_masks = targets['mask']
            seg_metrics = self.eval_segmentation(gt_masks, pred_masks)

        return {**base_metrics, **seg_metrics}


    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """
        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        log = {"val_loss": val_loss}

        all_keys = [*self.extra_metric_keys, *self.metric_keys]
        if self.use_mask:
            all_keys.extend(self.mask_metric_keys)

        for key in all_keys:
            log.update({f"val_{key}": weighted_mean(outs, f"val_{key}", "batch_size")})

        if not self.disable_knn_eval and not self.trainer.running_sanity_check:
            val_knn_acc1, val_knn_acc5 = self.knn.compute()
            log.update({"val_knn_acc1": val_knn_acc1, "val_knn_acc5": val_knn_acc5})

        self.log_dict(log, sync_dist=True)

    def eval_segmentation(self, gt_masks, pred_masks):
        # set up gt mask
        gt_seg_mask = torch.argmax(gt_masks, 1, True)

        # evaluate segmentation metrics
        new_ari, _ = average_ari(pred_masks, gt_seg_mask)
        new_ari_fg, _ = average_ari(pred_masks, gt_seg_mask, True)

        seg_mask = torch.argmax(torch.cat(pred_masks, 1), 1, True)
        msc, ssc, map, sap = average_segcover(gt_seg_mask.cpu(), seg_mask.cpu())
        msc_fg, ssc_fg, map_fg, sap_fg = \
            average_segcover(gt_seg_mask.cpu(), seg_mask.cpu(), ignore_background=True)
        device = seg_mask.device

        metrics = {
            "val_mask_ari": new_ari.to(device),
            "val_mask_ari_fg": new_ari_fg.to(device),
            "val_mask_msc": msc.to(device),
            "val_mask_msc_fg": msc_fg.to(device),
            "val_mask_ssc": ssc.to(device),
            "val_mask_ssc_fg": ssc_fg.to(device),
            "val_mask_map": map.to(device),
            "val_mask_map_fg": map_fg.to(device),
            "val_mask_sap": sap.to(device),
            "val_mask_sap_fg": sap_fg.to(device)
        }
        return metrics


class BaseMomentumADIOSModel(BaseMomentumModel):
    def __init__(self, dataset, train_mask_epoch=0, **kwargs):
        self.use_mask = (dataset in ["clevr", "coco"])
        super().__init__(**kwargs)
        # add summed, max metrics
        self.extra_metric_keys = []
        self.extra_metric_keys.extend([
            "mask_ari", "mask_ari_fg", "mask_msc", "mask_msc_fg", "mask_ssc", "mask_ssc_fg",
            "mask_map", "mask_sap", "mask_map_fg", "mask_sap_fg"])
        self.train_mask_epoch = train_mask_epoch

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds basic momentum arguments that are shared for all methods.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parent_parser = super(BaseMomentumADIOSModel, BaseMomentumADIOSModel).add_model_specific_args(
            parent_parser
        )
        parser = parent_parser.add_argument_group("base")
        parser.add_argument("--train_mask_epoch", default=10, type=int)

        return parent_parser


    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        # select optimizer
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam)")

        # create optimizer
        if isinstance(self.learnable_params, dict):
            # collect learnable parameters
            idxs_no_scheduler = [
                i for i, m in enumerate(self.learnable_params['inpainter']) if m.pop("static_lr", False)
            ]

            optimizer = [optimizer(
                self.learnable_params['inpainter'],
                lr=self.lr,
                weight_decay=self.weight_decay,
                **self.extra_optimizer_args,
            ),
            optimizer(
                    self.learnable_params['mask'],
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    **self.extra_optimizer_args,
                )]
        else:
            idxs_no_scheduler = [
                i for i, m in enumerate(self.learnable_params) if m.pop("static_lr", False)
            ]
            optimizer = [optimizer(
                self.learnable_params,
                lr=self.lr,
                weight_decay=self.weight_decay,
                **self.extra_optimizer_args,
            )]

        # optionally wrap with lars
        if self.lars:
            optimizer = [LARSWrapper(
                opt,
                eta=self.eta_lars,
                clip=self.grad_clip_lars,
                exclude_bias_n_norm=self.exclude_bias_n_norm,
            ) for opt in optimizer]

        if self.scheduler == "none":
            return optimizer # todo: might need some touch up due to new optimiser structure
        else:
            if self.scheduler == "warmup_cosine":
                scheduler = [
                    LinearWarmupCosineAnnealingLR(
                        optimizer[0],
                        warmup_epochs=self.warmup_epochs,
                        max_epochs=self.max_epochs-self.train_mask_epoch,
                        warmup_start_lr=self.warmup_start_lr,
                        eta_min=self.min_lr,
                    ),
                    LinearWarmupCosineAnnealingLR(
                        optimizer[1],
                        warmup_epochs=self.warmup_epochs,
                        max_epochs=self.max_epochs,
                        warmup_start_lr=self.warmup_start_lr,
                        eta_min=self.min_lr,
                    )]
            elif self.scheduler == "cosine":
                scheduler = [CosineAnnealingLR(optimizer[0], self.max_epochs-self.train_mask_epoch, eta_min=self.min_lr),
                             CosineAnnealingLR(optimizer[1], self.max_epochs, eta_min=self.min_lr)]
            elif self.scheduler == "step":
                scheduler = [MultiStepLR(opt, self.lr_decay_steps) for opt in optimizer]
            else:
                raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

            if idxs_no_scheduler:
                get_lr = scheduler[0].get_lr if isinstance(scheduler, list) else scheduler.get_lr
                partial_fn = partial(
                    static_lr,
                    get_lr=get_lr,
                    param_group_indexes=idxs_no_scheduler,
                    lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
                )
                if isinstance(scheduler, list):
                    scheduler[0].get_lr = partial_fn
                else:
                    scheduler.get_lr = partial_fn

            return optimizer, scheduler



    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        # call parent class to get results from standard metrics
        base_metrics = super().validation_step(batch, batch_idx)

        seg_metrics = {}
        if self.use_mask:
            # forward pass, get masks
            X, targets = batch
            with torch.no_grad():
                mask_enc_feat = self.mask_encoder(X)
                masks = self.mask_head(mask_enc_feat)
                pred_masks = torch.chunk(masks, masks.shape[1], dim=1)
            gt_masks = targets['mask']
            seg_metrics = self.eval_segmentation(gt_masks, pred_masks)

        return {**base_metrics[0], **seg_metrics}, base_metrics[1]


    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """
        if self.use_mask:
            parent_outs = [out[0] for out in outs]
            val_loss = weighted_mean(parent_outs, "val_loss", "batch_size")
            log = {"val_loss": val_loss}
            for key in [*self.extra_metric_keys, *self.metric_keys]:
                log.update({f"val_{key}": weighted_mean(parent_outs, f"val_{key}", "batch_size")})
            self.log_dict(log, sync_dist=True)

        else:
            super().validation_epoch_end(outs)


    def eval_segmentation(self, gt_masks, pred_masks):
        # set up gt mask
        gt_seg_mask = torch.argmax(gt_masks, 1, True)

        # evaluate segmentation metrics
        new_ari, _ = average_ari(pred_masks, gt_seg_mask)
        new_ari_fg, _ = average_ari(pred_masks, gt_seg_mask, True)

        seg_mask = torch.argmax(torch.cat(pred_masks, 1), 1, True)
        msc, ssc, map, sap = average_segcover(gt_seg_mask.cpu(), seg_mask.cpu())
        msc_fg, ssc_fg, map_fg, sap_fg = \
            average_segcover(gt_seg_mask.cpu(), seg_mask.cpu(), ignore_background=True)
        device = seg_mask.device

        metrics = {
            "val_mask_ari": new_ari.to(device),
            "val_mask_ari_fg": new_ari_fg.to(device),
            "val_mask_msc": msc.to(device),
            "val_mask_msc_fg": msc_fg.to(device),
            "val_mask_ssc": ssc.to(device),
            "val_mask_ssc_fg": ssc_fg.to(device),
            "val_mask_map": map.to(device),
            "val_mask_map_fg": map_fg.to(device),
            "val_mask_sap": sap.to(device),
            "val_mask_sap_fg": sap_fg.to(device)
        }
        return metrics

