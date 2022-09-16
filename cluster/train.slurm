#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='maf-train'
#SBATCH --mail-user=drigoni@math.unipd.it 		## user's email
#SBATCH --error=maf-train.err
#SBATCH --output=maf-train.out
#SBATCH --partition=allgroups
#SBATCH --ntasks=3
#SBATCH --exclusive								### cores
#SBATCH --mem=50G								### RAM
#SBATCH --time=3-12							    ### time
#SBATCH --gres=gpu:1							### request GPU with reserved policy


### ============ SOME PRINT COMMANDS
echo -n 'Date: '
date
echo -n 'Directory: '
pwd
echo -n 'Questo job viene eseguito sui seguenti nodi: '
echo ${SLURM_NODELIST}
echo ''
echo ''


### ============ COMMAND TO EXECUTE
# default paths'
SHAREDIR="/conf/shared-software/Singularity/CUDA/"
envPath="${HOME}/Programs/miniconda/envs/maf/bin"
progPath="${HOME}/repository/Multimodal-Alignment-Framework/"
cd $progPath

# input config file for the script
CMD=$1

python main.py

# last default print
echo ''
echo ''
echo 'Job done.'
echo -n 'Date: '
date
