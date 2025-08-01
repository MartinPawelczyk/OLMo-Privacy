#!/usr/bin/env bash
#SBATCH --job-name=olmo                     # Job name
#SBATCH -n 1                                # Number of cores
#SBATCH -N 1                                # Ensure that all cores are on one machine
#SBATCH -t 0-04:00                          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:2  # gpu:tesla_v100-pcie-32gb:1
#SBATCH -p gpu,seas_gpu
#SBATCH -v                              # make outout more verbose 
#SBATCH --cpus-per-task=5              # Number of CPU cores per task
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

# --- Recommended for the socket.cpp error ---
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=30000 # Use a different port if 29500 is known to be an issue

# --- ADD THESE LINES TO FORCE DISABLE W&B ---
# export WANDB_DISABLED="true"
# export WANDB_MODE="disabled" # Using both just to be super safe

echo "Starting OLMo training..."
WANDB_MODE=disabled WANDB_DISABLED=true \
torchrun \
 --master_addr="127.0.0.1" \
 --master_port="34567" \
 --nproc_per_node=2 \
 scripts/train.py configs/official-0425/OLMo2-1B-stage2-seed42.yaml