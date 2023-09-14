#!/bin/bash
# params
BATCH_SIZE=5
DEVICE='cuda'
COSINE_SIMILARITY_STRATEGY='mean'
LOSS_STRATEGY='ce'
MODEL_NAME='debug'
EPOCHS=2

# command
python main.py  --batch ${BATCH_SIZE} \
                                --device ${DEVICE} \
                                --debug \
                                --train_fract 0.1 \
                                --save_name ${MODEL_NAME} \
                                --cosine_similarity_strategy ${COSINE_SIMILARITY_STRATEGY} \
                                --loss_strategy ${LOSS_STRATEGY} \
                                --epochs ${EPOCHS} \
                                --emb_dim 300 \
                                --dataset 'referit' \
