python main_pretrain.py \
    --alpha_entropy 0 \
    --alpha_sparsity 0.39131043617986705 \
    --batch_size 128 \
    --classifier_lr 0.1 \
    --dataset stl10 \
    --lr 0.102523479861102 \
    --mask_lr 0.0941063568112331 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --weight_decay 0.0001 \
    --temperature 0.2 \
    --max_epochs 200 \
    --N 4 \
    --encoder vit_tiny \
    --mask_fbase 32 \
    --method simclr_adios \
    --output_dim 128 \
    --proj_hidden_dim 2048 \
    --unet_norm gn \
    --gpus 0 \
    --data_dir /datasets/yshi \
    --wandb_dir /datasets/yshi/adios \
    --checkpoint_dir /datasets/yshi/adios/trained_models \
    --project adios \
    --entity yugeten \
    --name simclr_adios_vit_stl10 \
    --wandb True
# note: replace --data_dir, --wandb_dir, --checkpoint_dir, --project, --entity, --name with your custom values.

