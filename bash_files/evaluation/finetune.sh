python main_finetune.py \
    --dataset stl10 \
    --data_dir /datasets/yshi \
    --max_epochs 200 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.5 \
    --weight_decay 5e-4 \
    --batch_size 128 \
    --num_workers 10 \
    --name finetune_imagenet100 \
    --pretrained_feature_extractor /datasets/yshi/adios/trained_models/simclr_adios/nzvrdc2h \
    --project adios \
    --entity yugeten \
    --wandb True
# change --pretrained_feature_extractor to where the model-to-evaluate is stored and --data_dir to your own data dir

