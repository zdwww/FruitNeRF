#!/bin/bash
#SBATCH -J 3d_seg
#SBATCH --mail-type=END
#SBATCH --time=48:00:00
#SBATCH --gres=gpumem:24g
#SBATCH --gpus=1
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=64G
PLOT=plot_461
#SBATCH --output=sbatch_log/run_${PLOT}.out
#SBATCH --error=sbatch_log/run_${PLOT}.err

module load stack/2024-04
module load gcc/8.5.0 cuda/11.8.0 ninja/1.11.1 ffmpeg/4.4.1 eth_proxy
echo $(module list)

source ~/.bashrc
conda activate base

conda activate nerfstudio

export PYTHONPATH=/cluster/home/daizhang/FruitNeRF
export WANDB_API_KEY=75a89a1a45f5525dc3717034484308953c5e267a

DATASET_PATH=/cluster/scratch/daizhang/Wheat-GS-data-nerf
WORKSPACE_PATH=/cluster/scratch/daizhang/FruitNeRF-workspace
DATE=20240717

TYPE=fruit_nerf

echo "****************** GPU Info ******************"
nvidia-smi

echo "****************** Start Training FruitNeRF ******************"
ns-train ${TYPE} \
    --data ${DATASET_PATH}/${DATE}/${PLOT} \
    --output-dir ${WORKSPACE_PATH}/output \
    --vis wandb \

