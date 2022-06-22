import os
from argparse import ArgumentParser, Namespace
from typing import Optional
from torchvision.utils import save_image

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

class AutoMASK(Callback):
    def __init__(
        self,
        args: Namespace,
        logdir: str,
        frequency: int = 1,
        keep_previous: bool = False,
        color_palette: str = "hls",
    ):
        """UMAP callback that automatically runs UMAP on the validation dataset and uploads the
        figure to wandb.

        Args:
            args (Namespace): namespace object containing at least an attribute name.
            logdir (str, optional): base directory to store checkpoints.
                Defaults to "auto_umap".
            frequency (int, optional): number of epochs between each UMAP. Defaults to 1.
            color_palette (str, optional): color scheme for the classes. Defaults to "hls".
            keep_previous (bool, optional): whether to keep previous plots or not.
                Defaults to False.
        """

        super().__init__()

        self.args = args
        self.logdir = logdir
        self.frequency = frequency
        self.color_palette = color_palette
        self.keep_previous = keep_previous

        self.colors = [(255, 255, 255), (0, 0, 255), (0, 255, 255), (255, 0, 255),
                  (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        self.colors.extend([(c[0] // 2, c[1] // 2, c[2] // 2) for c in self.colors])
        self.colors.extend([(c[0] // 4, c[1] // 4, c[2] // 4) for c in self.colors])
        self.colors = [torch.tensor(c) for c in self.colors]

    @staticmethod
    def add_auto_mask_args(parent_parser: ArgumentParser):
        """Adds user-required arguments to a parser.

        Args:
            parent_parser (ArgumentParser): parser to add new args to.
        """

        parser = parent_parser.add_argument_group("auto_mask")
        parser.add_argument("--auto_mask_dir", default="auto_mask", type=str)
        parser.add_argument("--auto_mask_frequency", default=1, type=int)
        return parent_parser

    def initial_setup(self, trainer: pl.Trainer):
        """Creates the directories and does the initial setup needed.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        if trainer.logger is None:
            version = None
        else:
            version = str(trainer.logger.version)
        if version is not None:
            self.path = os.path.join(self.logdir, trainer.logger.name, version)
        else:
            self.path = self.logdir

        self.umap_placeholder = "ep={}-n={}.png"
        self.last_ckpt: Optional[str] = None

        # create logging dirs
        if trainer.is_global_zero:
            os.makedirs(self.path, exist_ok=True)

    def on_train_start(self, trainer: pl.Trainer, _):
        """Performs initial setup on training start.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        self.initial_setup(trainer)

    def plot(self, trainer: pl.Trainer, module: pl.LightningModule):
        """Produces a UMAP visualization by forwarding all data of the
        first validation dataloader through the module.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
            module (pl.LightningModule): current module object.
        """
        if "none" not in self.path:
            device = module.device
            # set module to eval model and collect all feature representations
            module.eval()
            with torch.no_grad():
                for n, (x, _) in enumerate(trainer.val_dataloaders[0]):
                    x = x.to(device, non_blocking=True)[:8]
                    feats = module.mask_encoder(x)
                    soft_masks = module.mask_head(feats)
                    a = soft_masks.argmax(dim=1).cpu()
                    hard_masks = torch.zeros(soft_masks.shape).scatter(1, a.unsqueeze(1), 1.0)
                    save_tensor = [x.cpu()]
                    for mask in torch.chunk(hard_masks, self.args.N, dim=1):
                        save_tensor.extend([mask.repeat(1,3,1,1), x.cpu() * (1 - mask)])
                    path = os.path.join(self.path, self.umap_placeholder.format(trainer.current_epoch, n))
                    save_image(torch.cat(save_tensor), path)
            module.train()

    def on_validation_end(self, trainer: pl.Trainer, module: pl.LightningModule):
        """Tries to generate an up-to-date UMAP visualization of the features
        at the end of each validation epoch.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        epoch = trainer.current_epoch  # type: ignore
        if epoch % self.frequency == 0 and not trainer.sanity_checking:
            self.plot(trainer, module)
