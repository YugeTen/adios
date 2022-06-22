python main_pretrain.py \
    --alpha_entropy 0 \
    --alpha_sparsity 1 \
    --batch_size 128 \
    --dataset imagenet100 \
    --lr 0.8087906078374081 \
    --mask_lr 0.12952098519347935 \
    --max_epochs 400 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --weight_decay 1e-05\
    --brightness 0.5 \
    --contrast 0.2 \
    --hue 0.05 \
    --saturation 0.25 \
    --N 6 \
    --encoder resnet18 \
    --mask_fbase 32 \
    --method byol_adios_s \
    --output_dim 256 \
    --pred_hidden_dim 8192 \
    --proj_hidden_dim 4096 \
    --unet_norm gn \
    --gpus 0 \
    --data_dir /datasets/yshi \
    --wandb_dir /datasets/yshi/adios \
    --checkpoint_dir /datasets/yshi/adios/trained_models \
    --project adios \
    --entity yugeten \
    --name byol_adios_s_resnet18_imagenet100 \
    --wandb True
# note: replace --data_dir, --wandb_dir, --checkpoint_dir, --project, --entity, --name with your custom values.