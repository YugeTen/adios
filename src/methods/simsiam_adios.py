import argparse
from typing import Any, Dict, Sequence
import numpy as np

import torch
import torch.nn as nn
from src.losses.simsiam import simsiam_loss_func
from src.methods.base_adios import BaseADIOSModel
from src.utils.unet import UNet
from src.utils.blocks import str2bool
from src.utils.metrics import Entropy


class SimSiam_ADIOS(BaseADIOSModel):
    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        mask_lr: float,
        mask_fbase: int,
        N: int,
        num_blocks: int,
        alpha_sparsity: float,
        alpha_entropy: float,
        unet_norm: str,
        use_symmetry_mask: bool,
        use_no_mask: bool,
        use_both_mask: bool,
        **kwargs,
    ):
        """Implements Implements SimSiam (https://arxiv.org/abs/2011.10566) with ADIOS.
        In each forward pass, the masking function generates N masks from an input image,
        all of which are then applied to the image N times.

        Args:
            output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
            mask_lr (float): learning rate of masking model.
            alpha_sparsity (float): mask penalty term.
            alpha_entropy (float): mask penalty term 2
            N (int): number of masks to use.
            mask_fbase (int): base channels of masking model.
            unet_norm (str): normalisation function, choose from
                    - "no": no normalisation
                    - "in": instance noramlisation,
                    - "gn": group normalisation
            use_symmetry_mask (bool):  when set as true, it computes another loss term that
            makes the objective symmetrical by applying mask to the other augmented image.
            use_both_mask (bool): when set as true, it computes another loss term by applying
            mask to both augmented images.
            use_no_mask (bool):  when set as true, it computes another loss term without using
            mask on any of the augmented images.
        """
        super().__init__(**kwargs)

        # mask args
        self.alpha1 = alpha_sparsity
        self.alpha2 = alpha_entropy
        self.mask_lr = mask_lr
        self.mask_lr = self.mask_lr * self.accumulate_grad_batches
        self.N = N
        self.entropy = Entropy()

        # loss terms to compute
        self.use_both_mask = use_both_mask
        self.use_no_mask = use_no_mask
        self.use_symmetry_mask = use_symmetry_mask

        # simsiam projector
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

        # simsiam predictor
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, pred_hidden_dim, bias=False),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, output_dim),
        )

        # masking model
        self.mask_head = nn.Sequential(
            nn.Conv2d(mask_fbase, N, 1, 1, 0),
            nn.Softmax(dim=1)
        )

        self.mask_encoder = UNet(
            num_blocks=num_blocks,
            img_size=self.img_size,
            filter_start=mask_fbase,
            in_chnls=3,
            out_chnls=-1,
            norm=unet_norm)

        # Activates manual optimization
        self.automatic_optimization = False

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimSiam_ADIOS, SimSiam_ADIOS).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simsiam")
        # projector
        parser.add_argument("--output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=512)
        # mask args
        parser.add_argument("--mask_lr", type=float, default=0.1)
        parser.add_argument("--N", type=int, default=6)
        parser.add_argument("--num_blocks", type=int, default=4)
        parser.add_argument("--mask_fbase", type=int, default=32)
        parser.add_argument("--alpha_sparsity", type=float, default=1.)
        parser.add_argument("--alpha_entropy", type=float, default=0.)
        parser.add_argument("--unet_norm", type=str, default='gn',
                            choices=['gn', 'in', 'no'])
        # objective
        parser.add_argument("--use_symmetry_mask", type=str2bool, nargs='?',
                            const=True, default=True)
        parser.add_argument("--use_no_mask", type=str2bool, nargs='?',
                            const=True, default=True)
        parser.add_argument("--use_both_mask", type=str2bool, nargs='?',
                            const=True, default=True)
        return parent_parser

    @property
    def learnable_params(self) -> Dict[str, Any]:
        """Adds projector, predictor and masking model parameters to the
        parent's learnable parameters.

        Returns:
            Dict[str, Any]: dictionary of learnable parameters.
        """
        inpainter_learnable_params = \
            super().learnable_params + [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters(), "static_lr": True},
        ]
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
        for param in self.predictor.parameters():
            param.requires_grad = status

    def training_step(self, batch: Sequence[Any], batch_idx: int):
        """Training step for SimSiam ADIOS.

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

    def linear_forward(self, batch: Sequence[Any], batch_idx: int) -> Dict[str, Any]:
        """Forward function for masking model (occlusion model).

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
        """
        out = super().training_step(batch, batch_idx)
        return out["loss"]

    def mask_forward(self, batch: Sequence[Any]) -> torch.Tensor:
        """Forward function for masking model (occlusion model).

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
        """
        indexes, [x_orig, x_transformed], target = batch

        # generate masks
        masks = []
        for x in [x_orig, x_transformed]:
            enc_feat = self.mask_encoder(x)
            mask_feat = self.mask_head(enc_feat)
            masks.append(torch.chunk(mask_feat, self.N, dim=1))

        # forward pass for inference model
        similarities, summed_mask = [], []
        for k, (mask_orig, mask_transformed) in enumerate(zip(*masks)):
            z1, z2 = \
                self.projector(self.encoder(x_orig)), self.projector(self.encoder(x_transformed))
            z1_mask, z2_mask \
                = self.projector(self.encoder(x_orig*(1-mask_orig))), \
                  self.projector(self.encoder(x_transformed*(1-mask_transformed)))
            p1, p2, p1_mask, p2_mask = \
                self.predictor(z1), self.predictor(z2), \
                self.predictor(z1_mask), self.predictor(z2_mask)

            # compute objective
            loss = []
            # masked p, unmasked z
            loss.append(simsiam_loss_func(p1_mask, z2))
            loss.append(simsiam_loss_func(p2_mask, z1))
            # masked z, unmasked p
            if self.use_symmetry_mask:
                loss.append(simsiam_loss_func(p2, z1_mask))
                loss.append(simsiam_loss_func(p1, z2_mask))
            # no mask for z or p
            if self.use_no_mask:
                loss.append(simsiam_loss_func(p1, z2))
                loss.append(simsiam_loss_func(p2, z1))
            # use mask for z and p
            if self.use_both_mask:
                loss.append(simsiam_loss_func(p1_mask, z2_mask))
                loss.append(simsiam_loss_func(p2_mask, z1_mask))
            loss = torch.stack(loss).mean()

            # compute mask penalty
            for mask in [mask_orig, mask_transformed]:
                mask_penalty = mask.sum([-1, -2, -3]) / self.img_size**2
                summed_mask.append(mask_penalty)
                loss -= self.alpha1 * (1 / (torch.sin(mask_penalty * np.pi) + 1e-10)).mean(0).sum(0)
            similarities.append(loss)

        similarity = torch.stack(similarities).sum()
        all_summed_masks = torch.stack(summed_mask, dim=1)
        similarity += self.alpha2 * self.entropy(all_summed_masks)
        minval, _ = torch.stack(summed_mask).min(dim=0)
        maxval, _ = torch.stack(summed_mask).max(dim=0)
        self.log_dict({"train_mask_loss": similarity,
                       "mask_summed_min": minval.mean(),
                       "mask_summed_max": maxval.mean()},
                      on_epoch=True, sync_dist=True)
        return similarity

    def inpaint_forward(self, batch: Sequence[Any]) -> torch.Tensor:
        """Forward function for inpainter (inference model).

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
        """
        indexes, [x_orig, x_transformed], target = batch

        # generate masks
        masks = []
        for x in [x_orig, x_transformed]:
            enc_feat = self.mask_encoder(x)
            mask_feat = self.mask_head(enc_feat)
            masks.append(torch.chunk(mask_feat, self.N, dim=1))

        # forward pass for inference model
        similarities, summed_mask = [], []
        for k, (mask_orig, mask_transformed) in enumerate(zip(*masks)):
            mask_orig, mask_transformed = mask_orig.detach(), mask_transformed.detach()
            z1, z2 = self.projector(self.encoder(x_orig)), self.projector(self.encoder(x_transformed))
            z1_mask, z2_mask \
                = self.projector(self.encoder(x_orig*(1-mask_orig))), \
                  self.projector(self.encoder(x_transformed*(1-mask_transformed)))
            p1, p2, p1_mask, p2_mask = \
                self.predictor(z1), self.predictor(z2), self.predictor(z1_mask), self.predictor(z2_mask)

            # compute objective
            loss = []
            # masked p, unmasked z
            loss.append(simsiam_loss_func(p1_mask, z2))
            loss.append(simsiam_loss_func(p2_mask, z1))
            # masked z, unmasked p
            if self.use_symmetry_mask:
                loss.append(simsiam_loss_func(p2, z1_mask))
                loss.append(simsiam_loss_func(p1, z2_mask))
            # no mask for z or p
            if self.use_no_mask:
                loss.append(simsiam_loss_func(p1, z2))
                loss.append(simsiam_loss_func(p2, z1))
            # use mask for z and p
            if self.use_both_mask:
                loss.append(simsiam_loss_func(p1_mask, z2_mask))
                loss.append(simsiam_loss_func(p2_mask, z1_mask))
            loss = torch.stack(loss).mean()
            similarities.append(loss)

        similarity = torch.stack(similarities).sum()
        return similarity