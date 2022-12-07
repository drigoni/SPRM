#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='cosMean_lossLuca_epoch25_cosW'  # create a short name for your job
#SBATCH --mail-user=davide.rigoni.2@phd.unipd.it
#SBATCH --output=/home/lparolar/Projects/weakvg/out/flickr/cosMean_lossLuca_epoch25_cosW.out
#SBATCH --error=/home/lparolar/Projects/weakvg/out/flickr/cosMean_lossLuca_epoch25_cosW.err
#SBATCH --partition=allgroups
#SBATCH --mem=30G                                                   # total memory per node
#SBATCH --time=0-12:00:00                           # total run time limit (DD-HH:MM:SS)
#SBATCH --nodes=1                                   # node count
#SBATCH --ntasks-per-node=1                         # total number of tasks per node
#SBATCH --cpus-per-task=2                                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-node=1                           # number of gpus per node
#--- #SBATCH --array=1-3
#--- #SBATCH --exclusive
#--- #MORE COMMANDS
#----#SBATCH --mail-type=begin                           # send email when job begins
#--- #SBATCH --mail-type=end                             # send email when job ends
#--- #SBATCH --error=cluster/err/dynamicHead_train_scalable.err
#--- #SBATCH --ntasks=2
#--- #SBATCH --gpus-per-task=4
#---
#--- #SBATCH --gres=gpu:4                                # number of gpus per node


### ============ COMMAND TO EXECUTE
COSINE_WEIGHTS=(0 0.25 0.5 0.75 1)
# bash "./experiments/LSTM/cosMean_lossLuca_epoch25_cosW${COSINE_WEIGHTS[${SLURM_ARRAY_TASK_ID}]//.}.bash"
# params
ENV_PATH=/home/lparolar/Programs/miniconda/envs/weakvg/bin
PROJECT_HOME=/home/lparolar/Projects/weakvg
BATCH_SIZE=16
EPOCHS=25
COSINE_WEIGHT=0.5
###${COSINE_WEIGHTS[${SLURM_ARRAY_TASK_ID}]}
DEVICE='cuda'
COSINE_SIMILARITY_STRATEGY='mean'
LOSS_STRATEGY='luca'
MODEL_NAME='cosMean_lossCe_epoch25_cosW0'

# command
${ENV_PATH}/python ${PROJECT_HOME}/main.py  --batch ${BATCH_SIZE} \
                                --epochs ${EPOCHS} \
                                --cosine_weight ${COSINE_WEIGHT} \
                                --device ${DEVICE} \
                                --save_name ${MODEL_NAME} \
                                --cosine_similarity_strategy ${COSINE_SIMILARITY_STRATEGY} \
                                --loss_strategy ${LOSS_STRATEGY}


# last default print
echo ''
echo ''
echo 'Job done.'
echo 'Date: ' $(date)
