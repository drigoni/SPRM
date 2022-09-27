#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='cosMean_lossAll_epoch25_cosW05'  # create a short name for your job
#SBATCH --mail-user=davide.rigoni.2@phd.unipd.it
#SBATCH --output=cluster/out/cosMean_lossAll_epoch25_cosW05.out
#SBATCH --partition=testing
#SBATCH --mem=50G				                    # total memory per node
#SBATCH --time=2-00:00:00                           # total run time limit (DD-HH:MM:SS)
#SBATCH --nodes=1                                   # node count
#SBATCH --ntasks-per-node=1                         # total number of tasks per node
#SBATCH --cpus-per-task=4		                    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-node=1                           # number of gpus per node
#SBATCH --constraint=A5000
#SBATCH --exclusive								
#---- MORE COMMANDS
#----#SBATCH --mail-type=begin                           # send email when job begins
#--- #SBATCH --mail-type=end                             # send email when job ends
#--- #SBATCH --error=cluster/err/dynamicHead_train_scalable.err
#--- #SBATCH --ntasks=2
#--- #SBATCH --gpus-per-task=4
#--- #SBATCH --constraint=A5000
#--- #SBATCH --gres=gpu:4                                # number of gpus per node


### ============ VARIABLES SETTING
SLURM_MASTER_PORT=$(expr 10000 + $(echo -n ${SLURM_JOBID} | tail -c 4))
SLURM_MASTER_NODE="$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)" # such as gn11
SLURM_MASTER_ADDR="${SLURM_MASTER_NODE}"
SLURM_MASTER_URL="tcp://${SLURM_MASTER_ADDR}:${SLURM_MASTER_PORT}"
export SLURM_MASTER_PORT=SLURM_MASTER_PORT
export SLURM_MASTER_NODE=$SLURM_MASTER_NODE
export SLURM_MASTER_ADDR=$SLURM_MASTER_ADDR

### ============ SOME PRINT COMMANDS
echo ''
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo 'Date: ' $(date)
echo 'Directory: ' $(pwd)
echo "Nodelist: " ${SLURM_JOB_NODELIST}                  # like gn11 or gn[37,58]
echo "Number of nodes: " ${SLURM_JOB_NUM_NODES}
echo "Ntasks per node: "  ${SLURM_NTASKS_PER_NODE}
echo "NGPUs per node: "  ${SLURM_GPUS_PER_NODE}
echo "CUDA_VISIBLE_DEVICES: " ${CUDA_VISIBLE_DEVICES}
echo "TORCH_DEVICE_COUNT: " $(python -c 'import torch; print(torch.cuda.device_count())')
echo "SLURM_MASTER_PORT: " ${SLURM_MASTER_PORT}
echo "SLURM_MASTER_NODE: " ${SLURM_MASTER_NODE}
echo "SLURM_MASTER_ADDR: " ${SLURM_MASTER_ADDR}
echo "SLURM_MASTER_URL: " ${SLURM_MASTER_URL}
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo ''

### ============ COMMAND TO EXECUTE
bash ./experiments/LSTM/cosMean_lossCe_epoch25_cosW025.bash

# last default print
echo ''
echo ''
echo 'Job done.'
echo 'Date: ' $(date)
