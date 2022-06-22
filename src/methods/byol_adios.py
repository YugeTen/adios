import argparse
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np

import torch
import torch.nn as nn
from src.losses.byol import byol_loss_func
from src.methods.base_adios import BaseMomentumADIOSModel
from src.utils.momentum import initialize_momentum_params
from src.utils.unet import UNet
from src.utils.metrics import Entropy

class BYOL_ADIOS(BaseMomentumADIOSModel):
    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        mask_lr: float,
        mask_fbase: int,
        N: int,
        alpha_sparsity: float,
        alpha_entropy: float,
        unet_norm: str,
        **kwargs,
    ):
        """Implements BYOL (https://arxiv.org/abs/2006.07733) with ADIOS. In each forward pass,
        the masking function generates N masks from an input image, all of which are then applied
        to the image N times.

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
        """

        super().__init__(**kwargs)

        # mask args
        self.alpha1 = alpha_sparsity
        self.alpha2 = alpha_entropy
        self.mask_lr = mask_lr
        self.mask_lr = self.mask_lr * self.accumulate_grad_batches
        self.N = N
        self.entropy = Entropy()

        # byol projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        # byol momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # byol predictor
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, pred_hidden_dim),
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
        parent_parser = super(BYOL_ADIOS, BYOL_ADIOS).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("byol")
        # projector
        parser.add_argument("--output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        parser.add_argument("--pred_hidden_dim", type=int, default=512)
        # mask args
        parser.add_argument("--mask_lr", type=float, default=0.1)
        parser.add_argument("--N", type=int, default=6)
        parser.add_argument("--mask_fbase", type=int, default=32)
        parser.add_argument("--alpha_sparsity", type=float, default=1.)
        parser.add_argument("--alpha_entropy", type=float, default=0.)
        parser.add_argument("--unet_norm", type=str, default='gn',
                            choices=['gn', 'in', 'no'])
        return parent_parser

    @property
    def learnable_params(self) -> Dict[str, Any]:
        """Adds projector, predictor and masking model parameters to the
         parent's learnable parameters.

         Returns:
             Dict[str, Any]: dictionary of learnable parameters.
         """
        inpainter_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters()},
        ] + super().learnable_params
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

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs forward pass of the online encoder (encoder, projector and predictor).

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the logits of the head.
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
        sim = self.adios_forward(batch, 'inpaint')
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
        m_loss = self.adios_forward(batch, 'mask')
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

    def adios_forward(self, batch: Sequence[Any], forward_type: str='inpaint') -> torch.Tensor:
        """Forward function for adios.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
            forward_type (str): sets forward mode to inpaint for forward of inference model, and
                mask for forward of the masking/occlusion model.
        """
        indexes, [x_orig, x_transformed], target = batch

        # generate masks
        enc_feat = self.mask_encoder(x_transformed)
        masks = self.mask_head(enc_feat)

        # forward pass of inference model
        similarities, summed_mask = [], []
        for k, mask in enumerate(torch.chunk(masks, self.N, dim=1)):
            if forward_type == 'inpaint':
                mask = mask.detach()
            with torch.no_grad():
                feats1_momentum, feats2_momentum = \
                    self.momentum_encoder(x_orig), self.momentum_encoder(x_transformed*(1-mask))
                z1_momentum = self.momentum_projector(feats1_momentum)
                z2_momentum = self.momentum_projector(feats2_momentum)

            feats1, feats2 = self.encoder(x_orig), self.encoder(x_transformed*(1-mask))
            z1 = self.projector(feats1)
            z2 = self.projector(feats2)
            p1 = self.predictor(z1)
            p2 = self.predictor(z2)
            # compute objective
            loss = byol_loss_func(p1, z2_momentum) + byol_loss_func(p2, z1_momentum)
            # compute mask penalty
            if forward_type == 'mask':
                mask_penalty = mask.sum([-1, -2, -3]) / self.img_size**2
                summed_mask.append(mask_penalty)
                loss -= self.alpha1 * (1/(torch.sin(mask_penalty*np.pi)+1e-10)).mean()
            similarities.append(loss)
        similarity = torch.stack(similarities).sum()

        if forward_type == 'mask':
            all_summed_masks = torch.stack(summed_mask, dim=1)
            similarity += self.alpha2 * self.entropy(all_summed_masks)
            minval, _ = torch.stack(summed_mask).min(dim=0)
            maxval, _ = torch.stack(summed_mask).max(dim=0)
            self.log_dict({"train_mask_loss": similarity,
                           "mask_summed_min": minval.mean(),
                           "mask_summed_max": maxval.mean()},
                          on_epoch=True, sync_dist=True)
        return similarity
