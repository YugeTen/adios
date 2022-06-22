import argparse
from typing import Any, Dict, Sequence
import numpy as np

import torch
import torch.nn as nn
from src.losses.simclr import simclr_loss_func
from src.methods.base_adios import BaseADIOSModel
from src.utils.unet import UNet
from src.utils.metrics import Entropy


class SimCLR_ADIOS_S(BaseADIOSModel):
    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        temperature: float,
        mask_lr: float,
        alpha_sparsity: float,
        alpha_entropy: float,
        N: int,
        mask_fbase: int,
        unet_norm: str,
        **kwargs
    ):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709) with ADIOSs(ADIOS single).
        In each forward pass, the masking function generates N masks from an input image.
        We then randomly sample one mask to be applied to the image.

        Args:
            output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
            mask_lr (float): learning rate of masking model.
            alpha_sparsity (float): mask penalty term.
            alpha_entropy (float): mask penalty term 2
            N (int): number of masks to use.
            mask_fbase (int): base channels of masking model.
            unet_norm (str): normalisation function, choose from
                    - "no": no normalisation
                    - "in": instance noramlisation,
                    - "gn": group normalisation
        """

        super().__init__(**kwargs)

        self.temperature = temperature
        self.mask_lr = mask_lr
        self.mask_lr = self.mask_lr * self.accumulate_grad_batches
        self.alpha1 = alpha_sparsity
        self.alpha2 = alpha_entropy
        self.N = N
        self.entropy = Entropy()

        # simclr projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        # masking model
        self.mask_head = nn.Sequential(
            nn.Conv2d(mask_fbase, N, 1, 1, 0),
            nn.Softmax(dim=1)
        )

        self.mask_encoder = UNet(
            num_blocks=int(np.log2(self.img_size)-1),
            img_size=self.img_size,
            filter_start=mask_fbase,
            in_chnls=3,
            out_chnls=-1,
            norm=unet_norm)

        # Activates manual optimization
        self.automatic_optimization = False


    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimCLR_ADIOS_S, SimCLR_ADIOS_S).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simclr")
        # simclr args
        parser.add_argument("--output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        parser.add_argument("--mask_fbase", type=int, default=32)
        parser.add_argument("--temperature", type=float, default=0.1)
        # adios args
        parser.add_argument("--mask_lr", type=float, default=0.1)
        parser.add_argument("--alpha_sparsity", type=float, default=1.)
        parser.add_argument("--alpha_entropy", type=float, default=0.)
        parser.add_argument("--N", type=int, default=6)
        parser.add_argument("--unet_norm", type=str, default='no',
                            choices=['gn', 'in', 'no'])
        return parent_parser

    @property
    def learnable_params(self) -> Dict[str, Any]:
        """Adds projector and masking model parameters to the parent's learnable parameters.

         Returns:
             Dict[str, Any]: dictionary of learnable parameters.
         """
        inpainter_learnable_params = \
            super().learnable_params + [{"params": self.projector.parameters()}]
        mask_learnable_params = [
            {
                "name": "mask_encoder",
                "params": self.mask_encoder.parameters(),
                "lr": self.mask_lr
            },
            {
                "name": "mask_head",
                "params": self.mask_head.parameters(),
                "lr": self.mask_lr
            }
        ]
        return {"inpainter": inpainter_learnable_params, "mask": mask_learnable_params}

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

    def flip_inpainter_grad(self, status):
        """Sets requires_grad of inpainter (inference) model as True or False.

        Args:
            status (bool): determines whether requires_grad is True or False.
        """
        for param in self.encoder.parameters():
            param.requires_grad = status
        for param in self.projector.parameters():
            param.requires_grad = status
        for param in self.classifier.parameters():
            param.requires_grad = status

    def training_step(self, batch: Sequence[Any], batch_idx: int):
        """Training step for SimCLR ADIOS.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
            batch_idx (int): index of the batch.
        """
        # get optimiser and scheduler
        opt_i, opt_m = self.optimizers()
        sch_i, sch_m = self.lr_schedulers()

        # inpainter (inference model) forward
        self.flip_inpainter_grad(status=True)
        opt_i.zero_grad()
        sim = self.inpaint_forward(batch)
        linear_loss = self.linear_forward(batch, batch_idx)
        i_loss = linear_loss + sim
        self.manual_backward(i_loss) # maximise similarity
        opt_i.step()
        # update scheduler only at the end of each epoch
        if self.trainer.is_last_batch:
            sch_i.step()

        # masking model (occlusion model) forward
        self.flip_inpainter_grad(status=False)
        opt_m.zero_grad()
        m_loss = self.mask_forward(batch)
        self.manual_backward(-m_loss) # minimise similarity
        opt_m.step()
        # update scheduler only at the end of each epoch
        if self.trainer.is_last_batch:
            sch_m.step()


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
        enc_feat = self.mask_encoder(x_transformed)
        masks = self.mask_head(enc_feat)

        # randomly choose one mask for each sample in the batch
        chosen_mask = torch.randint(low=0, high=self.N, size=(self.batch_size, ))
        mask = torch.stack([masks[i, c, ...] for i, c in enumerate(chosen_mask)], dim=0).unsqueeze(1)
        feats1, feats2 = self.encoder(x_orig), self.encoder(x_transformed*(1-mask))
        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        # compute similarity between mask and no mask
        loss_func_kwargs = {"temperature": self.temperature, "pos_only": False}
        similarity = simclr_loss_func(z1, z2, **loss_func_kwargs)
        return similarity


    def mask_forward(self, batch: Sequence[Any]) -> torch.Tensor:
        """Forward function for masking model (occlusion model).

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
        """
        indexes, [x_orig, x_transformed], target = batch
        enc_feat = self.mask_encoder(x_transformed)
        masks = self.mask_head(enc_feat)

        # randomly choose one mask for each sample in the batch
        chosen_mask = torch.randint(low=0, high=self.N, size=(self.batch_size, ))
        mask = torch.stack([masks[i, c, ...] for i, c in enumerate(chosen_mask)], dim=0).unsqueeze(1)
        feats1, feats2 = self.encoder(x_orig), self.encoder(x_transformed*(1-mask))
        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        # compute similarity between mask and no mask
        loss_func_kwargs = {"temperature": self.temperature, "pos_only": False}
        loss = simclr_loss_func(z1, z2, **loss_func_kwargs)

        # compute mask penalty
        mask_penalty = masks.sum([-1, -2]) / self.img_size ** 2
        summed_mask = torch.chunk(mask_penalty.view(-1), self.N * self.batch_size, dim=0)
        loss -= self.alpha1 * (1 / (torch.sin(mask_penalty * np.pi) + 1e-10)).mean(0).sum(0)
        loss += self.alpha2 * self.entropy(mask_penalty)

        minval, _ = torch.stack(summed_mask).min(dim=0)
        maxval, _ = torch.stack(summed_mask).max(dim=0)
        self.log_dict({"train_mask_loss": loss,
                       "mask_summed_min": minval.mean(),
                       "mask_summed_max": maxval.mean()},
                      on_epoch=True, sync_dist=True)
        return loss
