#!/bin/bash

# params
PROJECT_HOME=/ceph/hpc/home/eudavider/repository/weakvg
BATCH_SIZE=8
EPOCHS=25
DEVICE='cuda'
MODEL_NAME='MATnet_model'

# command
python ${PROJECT_HOME}/main.py  --batch ${BATCH_SIZE} \
                                --epochs ${EPOCHS} \
                                --device ${DEVICE} \
                                --save_name ${MODEL_NAME} \
                                --loss_strategy "ce" \
                                --MATnet
