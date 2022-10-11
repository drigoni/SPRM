#!/bin/bash

# params
PROJECT_HOME=/ceph/hpc/home/eudavider/repository/weakvg
BATCH_SIZE=1
DEVICE='cuda'
COSINE_SIMILARITY_STRATEGY='mean'
LOSS_STRATEGY='luca'
MODEL_NAME='debug'
EPOCHS=2

# command
python ${PROJECT_HOME}/main.py  --batch ${BATCH_SIZE} \
                                --device ${DEVICE} \
                                --debug \
                                --save_name ${MODEL_NAME} \
                                --cosine_similarity_strategy ${COSINE_SIMILARITY_STRATEGY} \
                                --loss_strategy ${LOSS_STRATEGY} \
                                --epochs ${EPOCHS}
