#!/bin/bash

# params
PROJECT_HOME=/home/drigoni/repository/weakvg
BATCH_SIZE=16
EPOCHS=25
COSINE_WEIGHT=1
DEVICE='cuda'
COSINE_SIMILARITY_STRATEGY='mean'
LOSS_STRATEGY='ce'
MODEL_NAME='cosMean_lossCe_epoch25_cosW1'

# command
python ${PROJECT_HOME}/main.py  --batch ${BATCH_SIZE} \
                                --epochs ${EPOCHS} \
                                --cosine_weight ${COSINE_WEIGHT} \
                                --device ${DEVICE} \
                                --save_name ${MODEL_NAME} \
                                --cosine_similarity_strategy ${COSINE_SIMILARITY_STRATEGY} \
                                --loss_strategy ${LOSS_STRATEGY}
