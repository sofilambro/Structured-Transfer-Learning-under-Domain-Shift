#!/bin/bash
# =============================================================================
# TinyImageNet controlled F/T/S sweep -- SLURM array job.
#
# 58 jobs: 2 source networks + 14 partitions x 4 source/target directions.
# Unlike the iWildCam sweep this protocol is fixed by EPOCHS, not wall clock --
# every configuration trains for exactly 100 epochs, so no run is truncated for
# being slow and the comparison is fully controlled.
#
# ORDER MATTERS. Array tasks 0 and 1 train the two source networks (baseA and
# baseB); every other task loads one of them. Submit them first and let them
# finish, or the dependent tasks exit with "Source network not found".
#
#   sbatch --array=0-1  slurm/tinyimagenet_sweep.sbatch     # source networks
#   sbatch --array=2-57 slurm/tinyimagenet_sweep.sbatch     # transfer runs
#
# Or chain them in one submission:
#   BASE=$(sbatch --parsable --array=0-1 slurm/tinyimagenet_sweep.sbatch)
#   sbatch --dependency=afterok:$BASE --array=2-57 slurm/tinyimagenet_sweep.sbatch
#
# BEFORE SUBMITTING
#   1. Download TinyImageNet into $DATA_DIR/tiny-imagenet-200/
#   2. python scripts/tinyimagenet_prepare.py --stage all --data_dir $DATA_DIR
#   3. `python scripts/tinyimagenet_run.py --list` prints the index -> job map.
#
# NOTE: this pipeline is a reimplementation from the paper's Appendix A.1. The
# published Tables 4 and 5 came from a codebase not preserved in this repo.
# =============================================================================

#SBATCH --job-name=tin_fts
#SBATCH --output=logs/tinyimagenet_%A_%a.out
#SBATCH --error=logs/tinyimagenet_%A_%a.err
#SBATCH --array=0-57
#SBATCH --time=06:00:00              # WRN-28-10, 100 epochs on 50k 64x64 images
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail

# ── EDIT: dataset location ───────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-$SCRATCH/transfer_learning/data}"
REPO_DIR="${REPO_DIR:-$HOME/Transfer_Learning_AI_project}"

# ── Environment ──────────────────────────────────────────────────────────────
module load miniconda3
module load cuda/12.4
source ~/.bashrc
conda activate transfer_env

cd "$REPO_DIR"
mkdir -p logs

echo "=== Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

python scripts/tinyimagenet_run.py \
    --index      "$SLURM_ARRAY_TASK_ID" \
    --data_dir   "$DATA_DIR" \
    --max_epochs 100

echo "=== Task $SLURM_ARRAY_TASK_ID finished at $(date) ==="
