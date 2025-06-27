#!/usr/bin/env bash
#SBATCH --job-name=olmo                 # Job name
#SBATCH -n 1                            # Number of cores
#SBATCH -N 1                            # Ensure that all cores are on one machine
#SBATCH -t 0-01:00                      # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --gres=gpu:4                    # gpu:tesla_v100-pcie-32gb:1
#SBATCH -p gpu,seas_gpu
#SBATCH -v                              # make outout more verbose 
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --mem=80000
#SBATCH -o logs/myoutput_%j.out              # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e logs/myerrors_%j.err              # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=martin.pawelczyk.1@gmail.com
#SBATCH --account=sneel_lab

            
scontrol show job ${SLURM_JOB_ID}
nvidia-smi
# export NCCL_TIMEOUT=1800000
# export WANDB__SERVICE_WAIT=6000
# export OLMO_SHARED_FS=1

# Load required modules
echo "Loading modules..."
module load python/3.10.9-fasrc01
module load cuda/12.4.1-fasrc01 cudnn/9.5.1.17_cuda12-fasrc01
echo "Modules loaded."

# conda create -n olmo-2
echo "Activating conda environment..."
conda activate olmo-2
# pip install -e .[all]

# --- ADD THESE LINES TO FORCE DISABLE W&B ---
export WANDB_DISABLED="true"
export WANDB_MODE="disabled" # Using both just to be super safe

echo "Starting OLMo training..."
WANDB_MODE=disabled WANDB_DISABLED=true \ 
torchrun --nproc_per_node=4 --master_port=29501 scripts/train.py configs/official-0425/OLMo2-1B-stage1.yaml