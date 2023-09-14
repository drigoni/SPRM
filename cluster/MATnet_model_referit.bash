PROJECT_HOME=./
BATCH_SIZE=64
EPOCHS=25
DEVICE='cuda'
MODEL_NAME='MATnet_model'
# command
python ${PROJECT_HOME}/main.py  --batch ${BATCH_SIZE} \
                                --epochs ${EPOCHS} \
                                --device ${DEVICE} \
                                --save_name ${MODEL_NAME} \
                                --loss_strategy "ce" \
                                --MATnet \
                                --dataset "referit"