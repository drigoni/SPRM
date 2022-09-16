#!/bin/bash

# params
PROJECT_HOME=/home/drigoni/repository/weakvg
BATCH_SIZE=16
DEVICE='cuda'
COSINE_SIMILARITY_STRATEGY='max'
LOSS_STRATEGY='luca'
MODEL_NAME='cosMax_luca'

# command
python ${PROJECT_HOME}/main.py  --batch ${BATCH_SIZE} \
                                --device ${DEVICE} \
                                --save_name ${MODEL_NAME} \
                                --cosine_similarity_strategy ${COSINE_SIMILARITY_STRATEGY} \
                                --loss_strategy ${LOSS_STRATEGY}
