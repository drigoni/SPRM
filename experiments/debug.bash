#!/bin/bash

# params
PROJECT_HOME=/ceph/hpc/home/eudavider/repository/weakvg
BATCH_SIZE=2
DEVICE='cuda'
COSINE_SIMILARITY_STRATEGY='mean'
LOSS_STRATEGY='ce'
MODEL_NAME='debug'
EPOCHS=2

# command
python ${PROJECT_HOME}/main.py  --batch ${BATCH_SIZE} \
                                --device ${DEVICE} \
                                --debug \
                                --train_fract 0.01 \
                                --save_name ${MODEL_NAME} \
                                --cosine_similarity_strategy ${COSINE_SIMILARITY_STRATEGY} \
                                --loss_strategy ${LOSS_STRATEGY} \
                                --epochs ${EPOCHS} \
                                --emb_dim 300 \
