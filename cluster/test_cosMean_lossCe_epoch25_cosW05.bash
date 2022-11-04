#!/bin/bash

# params
PROJECT_HOME=/home/drigoni/repository/weakvg
BATCH_SIZE=16
EPOCHS=25
COSINE_WEIGHT=0.5
DEVICE='cuda'
COSINE_SIMILARITY_STRATEGY='mean'
LOSS_STRATEGY='ce'
MODEL_NAME='cosMean_lossCe_epoch25_cosW05'
MODEL_CHECKPOINT='./output/cosMean_lossCe_epoch25_cosW05_backup.pt'

# command
python ${PROJECT_HOME}/test.py  --batch ${BATCH_SIZE} \
                                --cosine_weight ${COSINE_WEIGHT} \
                                --device ${DEVICE} \
                                --cosine_similarity_strategy ${COSINE_SIMILARITY_STRATEGY} \
                                --loss_strategy ${LOSS_STRATEGY} \
                                --file ${MODEL_CHECKPOINT}
