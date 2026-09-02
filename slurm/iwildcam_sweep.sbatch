#!/bin/bash
# =============================================================================
# iWildCam F/T/S sweep -- SLURM array job, one task per configuration.
#
# The archived 21-run sweep was produced this way. Each task gets the same
# wall-clock training ceiling, which is what makes the runs comparable: no
# configuration can buy accuracy by simply being allowed to train longer.
#
# The trade-off is visible in the results -- T5S1 is the slowest partition at
# ~592 s/epoch and reached only 19.6 epochs inside the ceiling, while cheaper
# partitions completed 35-44. Widen --time and --max_time_min together if you
# want every configuration to finish the same number of epochs instead.
#
# BEFORE SUBMITTING
#   1. Point DATA_DIR at the directory holding iwildcam_v2.0/
#      (it must contain metadata.csv and train/).
#   2. Set the partition/account for your cluster.
#   3. Create the environment:  pip install -e ".[dev,flops]"
#
#   sbatch slurm/iwildcam_sweep.sbatch      # all 21
#   squeue -u $USER
#   python scripts/make_leaderboard.py      # once the array finishes
# =============================================================================

#SBATCH --job-name=iwildcam_fts
#SBATCH --output=logs/iwildcam_%A_%a.out
#SBATCH --error=logs/iwildcam_%A_%a.err
#SBATCH --array=0-20                 # 21 monotonic partitions; see --list
#SBATCH --time=04:00:00              # 185 min training + eval + margin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            # matches num_workers=4
#SBATCH --mem=32G
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
echo "=== Python: $(python --version) ==="

# --index selects the partition by position, so the array index maps 1:1 onto
# the 21 configurations. `python scripts/iwildcam_run.py --list` prints the map.
python scripts/iwildcam_run.py \
    --index         "$SLURM_ARRAY_TASK_ID" \
    --data_dir      "$DATA_DIR" \
    --dataset_mode  full \
    --max_epochs    100 \
    --max_time_min  185

echo "=== Task $SLURM_ARRAY_TASK_ID finished at $(date) ==="
