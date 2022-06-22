python main_knn.py \
    --dataset stl10 \
    --data_dir /datasets/yshi \
    --batch_size 128 \
    --num_workers 10 \
    --pretrained_feature_extractor /datasets/yshi/adios/trained_models/simclr_adios/nzvrdc2h \
    --k 1 2 5 10 20 50 100 200 \
    --temperature 0.01 0.02 0.05 0.07 0.1 0.2 0.5 1 \
    --feature_type backbone projector \
    --distance_function euclidean cosine
# change --pretrained_feature_extractor to where the model-to-evaluate is stored and --data_dir to your own data dir
