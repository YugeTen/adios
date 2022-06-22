import argparse
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from src.losses.simclr import simclr_loss_func
from src.methods.base import BaseModel
from src.utils.blocks import str2bool


class SimCLR_GT(BaseModel):
    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        temperature: float,
        N: int,
        random: bool,
        **kwargs
    ):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709) with ground truth object
        mask (available for CLEVR) with a few morphing options.

        Args:
            output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
            N (int): number of masks to use.
            random (bool): whether to shuffle the ground truth mask within the mini-batch
        """

        super().__init__(**kwargs)
        assert kwargs['dataset'] == "clevr", \
            f"This method uses ground-truth mask and is only implemented for CLEVR dataset."
        self.temperature = temperature
        self.N = N
        self.randomise = random

        # simclr projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )


    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimCLR_GT, SimCLR_GT).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simclr")

        # simclr args
        parser.add_argument("--output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        parser.add_argument("--temperature", type=float, default=0.1)

        # mask args
        parser.add_argument("--morph", type=str, default="none",
                            choices=["none", "erosion", "dilation", "box"])
        parser.add_argument("--N", type=int, default=6)
        parser.add_argument("--random", type=str2bool, nargs='?',
                            const=True, default=False)
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
        """Training step for SimCLR with ground truth mask.

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
        masks = target['mask'] # get ground truth mask
        masks = masks[:, :min(self.N, masks.shape[1]) , ...] # take N masks from ground truth mask
        similarities, all_masks = [], []

        rnd_idx = torch.randperm(x_orig.shape[0])
        for n, mask in enumerate(torch.chunk(masks, masks.shape[1], dim=1)):
            mask = mask[:,:,1,...] # take the mask of one view only
            # for the last mask slot, compute the complementary mask with
            # respect to all the masks used so far
            if n == self.N-1:
                canvas = torch.ones_like(mask)
                mask = canvas-torch.min(torch.stack(all_masks).sum(0), canvas)
            if self.randomise:
                mask = mask[rnd_idx]
            feats1, feats2 = self.encoder(x_orig), self.encoder(x_transformed*(1-mask))
            z1 = self.projector(feats1)
            z2 = self.projector(feats2)

            # compute similarity between mask and no mask
            loss_func_kwargs = {"temperature": self.temperature, "pos_only": False}
            similarities.append(simclr_loss_func(z1, z2, **loss_func_kwargs))
            all_masks.append(mask)

        similarity = torch.stack(similarities).sum()
        return similarity
