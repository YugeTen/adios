import os
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

import numpy as np
import random
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from src.utils.masking_schemes import segmask_to_box

L = 100000

class ClevrDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train', morph='none', load_masks=False):
        self.data_dir = f"{data_dir}/clevr"
        self.split = split
        self.transform = transform

        if not os.path.exists(f"{self.data_dir}/train_indices.pt"):
            indices = torch.randperm(L)
            torch.save(indices[:int(L * 0.7)], f"{self.data_dir}/train_indices.pt")
            torch.save(indices[int(L * 0.7):], f"{self.data_dir}/val_indices.pt")

        self.split_indices = torch.load(f"{self.data_dir}/{split}_indices.pt")
        shape = torch.load(f"{self.data_dir}/shape.pt")
        color = torch.load(f"{self.data_dir}/color.pt")
        self.color, self.shape = color[self.split_indices], shape[self.split_indices]

        self.C, self.S = len(self.color.unique())-1, len(self.shape.unique())-1
        self.n_objects = (self.color!=0).sum(-1)

        self.load_masks = load_masks
        kernel = np.ones((10, 10))
        if morph == "none":
            self.morph_fn = lambda x: x
        elif morph == "erosion":
            self.morph_fn = lambda x: torch.tensor(binary_erosion(x, structure=kernel)).float()
        elif morph == "dilation":
            self.morph_fn = lambda x: torch.tensor(binary_dilation(x, structure=kernel)).float()
        elif morph == "box":
            self.morph_fn = lambda m: segmask_to_box(m)

    def __len__(self):
        return len(self.split_indices)

    def seg_to_binary_mask(self, idx, seed, view=0):
        masks = []
        for i in range(11):
            mp = f"{self.data_dir}/masks/{self.split_indices[idx]}_{i}.png"
            m = Image.open(mp)
            random.seed(seed)
            torch.manual_seed(seed)
            mask_t = self.transform(m, no_color=True)[view] \
                if self.split == 'train' else self.transform(m)
            mask = torch.zeros_like(mask_t)
            mask[mask_t > 0.5] = 1.
            masks.append(self.morph_fn(mask[0]))
        final_mask = torch.stack(masks)[1:, ...]
        return final_mask

    def __getitem__(self, idx):
        imgpath = f"{self.data_dir}/images/{self.split_indices[idx]}.png"
        img = Image.open(imgpath)
        anno = {}

        seed = np.random.randint(0, 2 ** 32)
        random.seed(seed)
        torch.manual_seed(seed)
        # img = self.transform(img)
        img = self.transform(img, no_color=True) \
            if self.split == 'train' else self.transform(img)

        shape, color = self.shape[idx], self.color[idx]
        shape, color = shape[shape!=0], color[color!=0]
        label = color + self.C*(shape-1) -1
        one_hot = torch.zeros(self.C * self.S)
        one_hot[label.long()] = 1.
        anno['labels'] = one_hot

        if self.load_masks:
            if self.split == 'train':
                anno_masks = []
                for view in [0, 1]:
                    anno_masks.append(self.seg_to_binary_mask(idx, seed, view))
                anno['mask'] = torch.stack(anno_masks, 1)
            else:
                anno['mask'] = self.seg_to_binary_mask(idx, seed)
            anno['n_objects'] = self.n_objects[idx]

        return img, anno

