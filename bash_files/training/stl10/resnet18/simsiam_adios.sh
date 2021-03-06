python main_pretrain.py \
    --batch_size 128 \
    --dataset stl10 \
    --lr 0.4436168048573669 \
    --mask_lr 0.038296424016559166 \
    --optimizer sgd \
    --weight_decay 7.227625739330174e-06 \
    --scheduler warmup_cosine \
    --alpha_entropy 0. \
    --alpha_sparsity 1. \
    --use_both_mask false \
    --use_no_mask true \
    --use_symmetry_mask false \
    --max_epochs 400 \
    --brightness 0.5521874334305088 \
    --contrast 0.4775521429505951 \
    --hue 0.2 \
    --saturation 0.5784474905390056 \
    --N 6 \
    --encoder resnet18 \
    --mask_fbase 32 \
    --method simsiam_adios \
    --output_dim 256 \
    --proj_hidden_dim 2048 \
    --pred_hidden_dim 512 \
    --unet_norm gn \
    --num_blocks 5 \
    --gpus 0 \
    --data_dir /datasets/yshi \
    --wandb_dir /datasets/yshi/adios \
    --checkpoint_dir /datasets/yshi/adios/trained_models \
    --project adios \
    --entity yugeten \
    --name simsiam_adios_resnet18_stl10 \
    --wandb True
# note: replace --data_dir, --wandb_dir, --checkpoint_dir, --project, --entity, --name with your custom values.
