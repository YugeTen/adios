python main_linear.py \
    --dataset stl10 \
    --data_dir /datasets/yshi \
    --max_epochs 100 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 128 \
    --num_workers 10 \
    --name linear_imagenet100 \
    --pretrained_feature_extractor /datasets/yshi/adios/trained_models/simclr_adios/nzvrdc2h \
    --project adios \
    --entity yugeten \
    --wandb True
# change --pretrained_feature_extractor to where the model-to-evaluate is stored and --data_dir to your own data dir

