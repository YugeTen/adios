# Adversarial Masking for Self-Supervised Learning
<img src="https://i.imgur.com/yGtZ9Gy.png" width="250" align="right">

This is the official PyTorch implementation for [Adversarial Masking for Self-Supervised Learning](https://arxiv.org/pdf/2201.13100). We propose ADIOS, a masked image modeling (MIM) framework for self-supervised learning, which simultaneously learns a masking function and an image encoder using an adversarial objective. 

Our code is developed on the amazing [solo-learn](https://github.com/vturrisi/solo-learn) repository, a library for self-supervised learning methods. We make use of [pytorch-lightning](https://www.pytorchlightning.ai/) for training pipeline and [wandb](https://wandb.ai/) for tracking experiments.






# Installing packages

First clone the repo. Install required packages by
```
pip3 install .
```


To install with [Dali](https://github.com/NVIDIA/DALI) and/or UMAP support, use:
```
pip3 install .[dali,umap] --extra-index-url https://developer.download.nvidia.com/compute/redist
```
The solo-learn repository has more [instructions](https://github.com/vturrisi/solo-learn#installation) on troubleshooting installing Dali.

## Requirements
* torch
* torchvision
* tqdm
* einops
* wandb
* pytorch-lightning
* lightning-bolts
* torchmetrics
* scipy
* timm

**Optional**:
* nvidia-dali
* matplotlib
* seaborn
* pandas
* umap-learn



# Downloading datasets

CIFAR10, CIFAR100 and STL10 datasets are automatically downloaded by running the code. 

## ImageNet100
First, download the ILSVRC 2012 ImageNet dataset from the [official website](https://image-net.org/index.php) to `<your_data_dir>/imagenet/` (this HAS to be done manually). Then, unzip both .tar files:

```
cd <your_data_dir>/imagenet
mkdir val
mkdir train
tar -xf ILSVRC2012_img_val.tar --directory val
tar -xf ILSVRC2012_img_train.tar --directory train
cd train
for f in *.tar; do mkdir -p "${f:0:9}" && tar -xvf "$f" -C "${f:0:9}"; done
```

After all files are unzipped, do the following:

```
cp <your_code_dir>/adios/bash_files/imagenet.sh <your_data_dir>/imagenet/val/
cd <your_data_dir>/imagenet/val/
./imagenet.sh
```
This will set up the validation set properly.

After setting up the ImageNet dataset, you can do the following to create the ImageNet100 dataset:

```
cp <your_code_dir>/adios/bash_files/imagenet100.sh <your_data_dir>/
cd <your_data_dir>
./imagenet100.sh
```

## CLEVR
Download from [here](https://www.robots.ox.ac.uk/~yshi/adios/clevr.zip), then unzip in `<your_data_dir>`.



# Training

## Launching a single experiment
You can find a range of example bash scripts with our selected hyperparameters in `bash_files/training`. The scripts are categorised by datasets, and then architecture. For example, to train a `BYOL_ADIOS` model from scratch on ImageNet100S dataset using ResNet18 backbone, you can run the following:

```bash
./bash_files/training/imagenet100s/resnet18/byol_adios.sh
```

Some args related to mask generation to pay attention to are:
- `--N`: sepcifies the number of masking slots
- `--mask_lr`: learning rate for the masking model
- `--alpha_sparsity`: sparsity penalty for masking model, see "sparsity penalty" in section 2.3 of our [paper](https://arxiv.org/pdf/2201.13100) for detail
- `--mask_fbase`: base channels of masking model
- `--unet_norm`: normalisation function for masking model, choose from "gn" (group normalisation), "in" (instance normalisation) and "no" (no normalisation).
- `--auto_mask`: set as true to automatically save generated masks every epoch
- `--pretrained_dir`: path to pretrained model (if any), set as None to load no pretrained model


## Pre-trained models
We offer the following downloadable pre-trained models:
### ImageNet100S
#### ResNet18 backbone
- [SimCLR_ADIOS](https://www.robots.ox.ac.uk/~yshi/adios/pretrained_models/imagenet100s/simclr_adios_imagenet100s.zip)
- [SimSiam_ADIOS](https://www.robots.ox.ac.uk/~yshi/adios/pretrained_models/imagenet100s/simsiam_adios_imagenet100s.zip)
- [BYOL_ADIOS](https://www.robots.ox.ac.uk/~yshi/adios/pretrained_models/imagenet100s/byol_adios_imagenet100s.zip)
#### Vision Transformer (tiny) backbone
- [SimCLR_ADIOS](https://www.robots.ox.ac.uk/~yshi/adios/pretrained_models/imagenet100s/simclr_adios_imagenet100s_vit.zip)
- [SimSiam_ADIOS](https://www.robots.ox.ac.uk/~yshi/adios/pretrained_models/imagenet100s/simsiam_adios_imagenet100s_vit.zip)
- [BYOL_ADIOS](https://www.robots.ox.ac.uk/~yshi/adios/pretrained_models/imagenet100s/byol_adios_imagenet100s_vit.zip)
### ImageNet100
For ImageNet100 we provide pretrained models of the single-forward pass version of ADIOS (ADIOS-s) trained using ResNet18 backbone.
- [SimCLR_ADIOS_S](https://www.robots.ox.ac.uk/~yshi/adios/pretrained_models/imagenet100/simclr_adioss_imagenet100.zip)
- [SimSiam_ADIOS_S](https://www.robots.ox.ac.uk/~yshi/adios/pretrained_models/imagenet100/simsiam_adioss_imagenet100.zip)
- [BYOL_ADIOS_S](https://www.robots.ox.ac.uk/~yshi/adios/pretrained_models/imagenet100/byol_adioss_imagenet100.zip)

To load and run the pre-trained models, specify the path to the downloaded models using `--pretrained_dir` and make sure to use the same `--method` as the one used in the pre-trained model. You can run the model on any dataset you desire. For instance, to use the `BYOL_ADIOS` model trained on ImageNet100S with ResNet18 backbone, download the model from [here](https://www.robots.ox.ac.uk/~yshi/adios/pretrained_models/imagenet100s/byol_adios_imagenet100s.zip) and run:
```bash
python main_pretrain.py \
    --pretrained_dir <download_dir>/byol_adios_imagenet100s/whnjyonq \
    --method byol_adios \
    --dataset imagenet100s \
    --data_dir <your_data_dir> \
```

## Create a sweep

We make use of the sweep function in wandb for hyperparameter search. Here we include a small tutorial for how to do this in this repository. For further detail please refer to their [documentations](https://wandb.ai/site/sweeps). 
You can find an example sweep yaml files in `bash_files/sweep/example.yaml`.


After creating an account with wandb, login in your terminal by:

```
wandb login
```
Then follow the insutrctions on screen.

After logging in, you can do the following to initiate a sweep:

```
wandb sweep bash_files/sweep/example.yaml
```

Then you will see something like this:

```
wandb: Creating sweep from: example.yaml
wandb: Created sweep with ID: 0y3kc50c
wandb: View sweep at: https://wandb.ai/yugeten/imagenet/sweeps/0y3kc50c
wandb: Run sweep agent with: wandb agent yugeten/imagenet/0y3kc50c
```

Then to initiate one run for the sweep, you can simply do:

```
CUDA_VISIBLE_DEVICES=X wandb agent yugeten/imagenet100/0y3kc50c
```

After the current run finish, another fresh run will start automatically on the specified GPU.

# Evaluation

In `bash_files/evaluation` you can find several example bash scripts that evaluates trained model. We offer linear evaluation (`linear.sh`), knn evaluation (`knn.sh`), finetuning (`finetune.sh`) and FMI, ARI, NMI clustering (`clustering.sh`).




# Citation
If you found our work helpful, please cite us:
```
@inproceedings{shi2022adversarial,
  title={Adversarial Masking for Self-Supervised Learning},
  author={Shi, Yuge and Siddharth, N and Torr, Philip HS and Kosiorek, Adam R},
  booktitle = {International Conference on Machine Learning},
  year={2022}
}
```
