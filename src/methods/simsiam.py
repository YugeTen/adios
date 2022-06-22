import argparse, os, json
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.simsiam import simsiam_loss_func
from src.methods.base import BaseModel


class SimSiam(BaseModel):
    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        pre_trained: str,
        **kwargs,
    ):
        """Implements SimSiam (https://arxiv.org/abs/2011.10566).

        Args:
            output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """

        super().__init__(**kwargs)

        self.load_pretrained = (pre_trained!="")
        # possibly change hps to match the pretrained model
        if pre_trained != "" and os.path.exists(pre_trained):
            files = list(os.walk(pre_trained))[0][-1]
            files.sort()
            p_args_file, p_model_file = files
            p_args_path, p_model_path = os.path.join(pre_trained, p_args_file), os.path.join(pre_trained, p_model_file)
            with open(p_args_path, "r") as file:
                p_args = json.load(file) # load pretrained args
                proj_hidden_dim, pred_hidden_dim, output_dim = \
                    p_args['proj_hidden_dim'], p_args['pred_hidden_dim'], p_args['output_dim']

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False),
        )
        self.projector[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, pred_hidden_dim, bias=False),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, output_dim),
        )

        # load pretrained model, if appropriate
        if pre_trained != "" and os.path.exists(pre_trained):
            assert p_args['encoder'] == kwargs['encoder'], \
                f"Backbone mismatch! Pretrained path is specified as {pre_trained} " \
                f"with backbone {p_args['encoder']}, but the model's encoder is set " \
                f"as {kwargs['encoder']}. Please change your setting."
            # load model
            pretrained_model = torch.load(p_model_path, map_location="cpu")
            self.load_state_dict(pretrained_model['state_dict'], strict=False)
            print(f"\n\n\nLoaded pretrained model from {p_model_path}\n\n\n")
            # if load_pretrained_optim:
            #     self.pretrained_optims = pretrained_model['optimizer_states'][0]
            self.pretrained_optims = pretrained_model['optimizer_states'][0]
            self.pretrained_schedulers = pretrained_model['lr_schedulers'][0]
            self.loaded_optim_scheduler_state = False

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimSiam, SimSiam).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simsiam")

        # projector
        parser.add_argument("--output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=512)
        parser.add_argument("--pre_trained", type=str, default="",
                            help="path to pretrained dir, leave empty for training from scratch")
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params: List[dict] = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters(), "static_lr": True},
        ]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the encoder, the projector and the predictor.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected and predicted features.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        return {**out, "z": z, "p": p}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for SimSiam reusing BaseModel training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images
            batch_idx (int): index of the batch

        Returns:
            Dict[str, Any]: total loss composed of SimSiam loss and classification loss
        """

        if self.load_pretrained and not self.loaded_optim_scheduler_state:
            self.lr_schedulers().load_state_dict(self.pretrained_schedulers)
            self.optimizers().load_state_dict(self.pretrained_optims)
            self.loaded_optim_scheduler_state = True

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats1, feats2 = out["feats"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # ------- contrastive loss -------
        neg_cos_sim = simsiam_loss_func(p1, z2) / 2 + simsiam_loss_func(p2, z1) / 2

        # calculate std of features
        z1_std = F.normalize(z1, dim=-1).std(dim=0).mean()
        z2_std = F.normalize(z2, dim=-1).std(dim=0).mean()
        z_std = (z1_std + z2_std) / 2

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss
