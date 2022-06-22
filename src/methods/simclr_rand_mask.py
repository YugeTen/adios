import argparse
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from src.losses.simclr import simclr_loss_func
from src.methods.base import BaseModel
from src.utils.masking_schemes import gen_mae_mask, gen_beit_mask

class SimCLR_RM(BaseModel):
    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        temperature: float,
        rand_mask_type: str,
        mask_perc: float,
        **kwargs
    ):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).
        Args:
            output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
            rand_mask_type (str): random mask generation scheme, following either
            MAE (https://arxiv.org/abs/2111.06377) or BEiT (https://arxiv.org/abs/2106.08254).
            mask_perc (float): percentage of the image to mask.
        """

        super().__init__(**kwargs)

        self.temperature = temperature

        # simclr projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        # masks
        self.mask_function = gen_mae_mask if rand_mask_type == "mae" else gen_beit_mask
        self.mask_kwargs = {} if mask_perc==-1. else {"mask_perc": mask_perc}

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimCLR_RM, SimCLR_RM).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simclr")

        # simclr args
        parser.add_argument("--output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        parser.add_argument("--temperature", type=float, default=0.1)
        # mask args
        parser.add_argument("--rand_mask_type", type=str, default="beit",
                            choices=["mae", "beit"])
        parser.add_argument("--mask_perc", type=float, default=-1.)
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.
        Returns:
            List[dict]: list of learnable parameters.
        """
        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor, *args, **kwargs) -> Dict[str, Any]:
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
        return {**out, "z": z}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR with random mask.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
            batch_idx (int): index of the batch.
        """
        sim = self.inpaint_forward(batch)
        linear_loss = self.linear_forward(batch, batch_idx)
        i_loss = linear_loss + sim
        return i_loss

    def linear_forward(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Forward function for linear classifier (backbone model is detached).

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
            batch_idx (int): index of the batch.
        """
        indexes, *_, target = batch
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        return class_loss

    def inpaint_forward(self, batch: Sequence[Any]) -> torch.Tensor:
        """Forward function for inpainter (inference model).

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
        """
        indexes, [x_orig, x_transformed], target = batch
        # generate random mask
        masks = torch.stack([self.mask_function(x_orig_[0,...], **self.mask_kwargs)
                             for x_orig_ in x_orig]).unsqueeze(1)

        feats1, feats2 = self.encoder(x_orig), self.encoder(x_transformed*(1-masks))
        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        # compute similarity between mask and no mask
        loss_func_kwargs = {"temperature": self.temperature, "pos_only": False}
        similarity = simclr_loss_func(z1, z2, **loss_func_kwargs)
        return similarity