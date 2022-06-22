python main_clustering.py \
    --dataset stl10 \
    --batch_size 16 \
    --num_workers 10 \
    --pretrained_feature_extractor /datasets/yshi/adios/trained_models/simclr_adios/nzvrdc2h \
    --feature_type backbone projector \
    --data_dir /datasets/yshi
# change --pretrained_feature_extractor to where the model-to-evaluate is stored and --data_dir to your own data dir
