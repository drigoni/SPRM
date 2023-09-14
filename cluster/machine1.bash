# params
PROJECT_HOME=./

# command
python ${PROJECT_HOME}/main.py  --dataset flickr30k \
                                --batch 16 \
                                --epochs 25 \
                                --cosine_weight 0.4 \
                                --device cuda \
                                --save_name data_augmentation_flickr30k_025 \
                                --cosine_similarity_strategy mean \
                                --loss_strategy luca_max \
                                --train_fract 1 \
                                --do_oov \
                                --do_head \
                                --use_head_for_concept_embedding \
                                # --test_set \