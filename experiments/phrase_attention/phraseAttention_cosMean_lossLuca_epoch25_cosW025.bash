#!/bin/bash

# params
PROJECT_HOME=/home/drigoni/repository/weakvg
BATCH_SIZE=8
EPOCHS=25
COSINE_WEIGHT=0.25
DEVICE='cuda'
COSINE_SIMILARITY_STRATEGY='mean'
LOSS_STRATEGY='luca'
MODEL_NAME='phraseAttention_cosMean_lossLuca_epoch25_cosW025'

# command
python ${PROJECT_HOME}/main.py  --batch ${BATCH_SIZE} \
                                --epochs ${EPOCHS} \
                                --cosine_weight ${COSINE_WEIGHT} \
                                --device ${DEVICE} \
                                --save_name ${MODEL_NAME} \
                                --cosine_similarity_strategy ${COSINE_SIMILARITY_STRATEGY} \
                                --loss_strategy ${LOSS_STRATEGY} \
                                --use_att_for_query
