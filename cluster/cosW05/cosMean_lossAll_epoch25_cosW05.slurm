#!/bin/bash
### Owner: Davide Rigoni

### ============ SLURM COMMANDS
#SBATCH --job-name='cosMean_lossAll_epoch25_cosW05'
#SBATCH --mail-user=drigoni@math.unipd.it 		## user's email
#SBATCH --error=cluster/err/cosMean_lossAll_epoch25_cosW05.err
#SBATCH --output=cluster/out/cosMean_lossAll_epoch25_cosW05.out
#SBATCH --partition=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=A5000
#SBATCH --exclusive								
#SBATCH --mem=50G								### RAM
#SBATCH --time=1-12:00:00					    ### time
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
bash ./experiments/cosW05/cosMean_lossAll_epoch25_cosW05.bash

# last default print
echo ''
echo ''
echo 'Job done.'
echo -n 'Date: '
date
