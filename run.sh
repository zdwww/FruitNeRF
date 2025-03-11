#!/bin/bash
#SBATCH -J fruit
#SBATCH --mail-type=END
#SBATCH --time=48:00:00
#SBATCH --gres=gpumem:24g
#SBATCH --gpus=1
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=32G
#SBATCH --output=sbatch_log/run_462.out
#SBATCH --error=sbatch_log/run_462.out

module load stack/2024-04
module load gcc/8.5.0 cuda/11.8.0 ninja/1.11.1 ffmpeg/4.4.1 eth_proxy
echo $(module list)

source ~/.bashrc
conda activate base

conda activate fruitnerf

export PYTHONPATH=/cluster/home/daizhang/FruitNeRF
export WANDB_API_KEY=75a89a1a45f5525dc3717034484308953c5e267a

DATASET_PATH=/cluster/scratch/daizhang/Wheat-GS-data-nerf
WORKSPACE_PATH=/cluster/scratch/daizhang/FruitNeRF-workspace
DATE=20240717
PLOT=plot_461

TYPE=fruit_nerf

echo "****************** GPU Info ******************"
nvidia-smi

echo "****************** Start Training FruitNeRF ******************"
ns-train ${TYPE} \
    --data ${DATASET_PATH}/${DATE}/${PLOT} \
    --output-dir ${WORKSPACE_PATH}/output \
    --vis wandb \
    pipeline:fruit-pipeline-config --pipeline.datamanager.camera-optimizer.mode off

# TIMESTAMP=$(ls -d ${WORKSPACE_PATH}/output/${PLOT}/${TYPE}/2025-03* | sort | tail -n 1 | awk -F'/' '{print $NF}')

# echo "*********** Start evaluation on largest TIMESTAMP $TIMESTAMP ***********"

# ns-export-semantics semantic-pointcloud \
#     --load-config ${WORKSPACE_PATH}/output/${PLOT}/${TYPE}/${TIMESTAMP}/config.yml \
#     --output-dir ${WORKSPACE_PATH}/output/${PLOT} \
#     --use-bounding-box True \
#     --bounding-box-min -1 -1 -1 \
#     --bounding-box-max 1 1 1 \
#     --num_rays_per_batch 2000 \
#     --num_points_per_side 2000

# for i in {461..467}; do
#     PLOT="plot_$i"
#     for TYPE in "fruit_nerf"; do
#         TIMESTAMP=$(ls -d ${WORKSPACE_PATH}/output/${PLOT}/${TYPE}/2025-03* | sort | tail -n 1 | awk -F'/' '{print $NF}')
#         echo "PLOT=$PLOT TYPE=$TYPE TIMESTAMP=$TIMESTAMP"
#         ns-eval-fruit \
#             --load-config ${WORKSPACE_PATH}/output/${PLOT}/${TYPE}/${TIMESTAMP}/config.yml \
#             --output-path ${WORKSPACE_PATH}/output/${PLOT}/${TYPE}/output.json \
#             --render-output-path ${WORKSPACE_PATH}/output/${PLOT}/${TYPE}/render
#         ns-export-semantics semantic-pointcloud \
#             --load-config ${WORKSPACE_PATH}/output/${PLOT}/${TYPE}/${TIMESTAMP}/config.yml \
#             --output-dir ${WORKSPACE_PATH}/output/${PLOT} \
#             --use-bounding-box True \
#             --bounding-box-min -1 -1 -1 \
#             --bounding-box-max 1 1 1 \
#             --num_rays_per_batch 2000 \
#             --num_points_per_side 2000
#     done
# done