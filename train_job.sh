#!/bin/bash
#SBATCH --job-name=dogs_vs_cats_train
#SBATCH --output=output-%j.log
#SBATCH --error=output-%j.log
#SBATCH --time=03:00:00
#SBATCH --gpus=v100:1

# Load required modules
. /cluster/apps/software/lmod/lmod/init/bash
module load CUDA/12.4.0
module load Miniconda3/25.5.1-0

# Activate conda environment
source activate
conda activate agent

# Change to project directory
cd /home/ee6483_40/Project2_Classification

# Run training
python -m src.engine.trainer \
  --cfg configs/base.yaml \
  --cfg configs/task/dogs_vs_cats.yaml \
  --cfg configs/model/densenet121.yaml \
  --cfg configs/aug/light.yaml \
  --cfg configs/optim/adamw.yaml

echo "Training completed!"
