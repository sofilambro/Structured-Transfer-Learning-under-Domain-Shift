#!/bin/bash
# =============================================================================
# iWildCam T5S1 timing benchmark — SLURM submission script
#
# BEFORE SUBMITTING:
#   1. Set DATA_DIR to wherever you uploaded iwildcam_v2.0/ on the cluster.
#      The directory must contain: metadata.csv  and  train/ (image folder).
#   2. Pick a partition (see options below) and uncomment accordingly.
#   3. Submit: sbatch run_hpc.sh
#   4. Monitor: squeue -u $USER
#   5. Download results: scp -r <user>@slogin... :~/HPC_EXPORT/results ./HPC_IMPORT/
# =============================================================================

#SBATCH --job-name=iwildcam_t5s1
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --time=01:30:00          # 90 min wall limit (45 min train + ~30 min eval + margin)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        # matches num_workers=4 in config
#SBATCH --mem=32G                # ResNet50 + full dataset metadata + image cache
#SBATCH --gres=gpu:1

# ── Partition — uncomment ONE block ──────────────────────────────────────────
# Option A: H200 (fastest, often idle — best for timing benchmark)
#SBATCH --partition=gpuh200

# Option B: L40 (also fast, often idle)
## #SBATCH --partition=gpul40

# Option C: Student partition (guaranteed access, older GPU)
## #SBATCH --partition=stud
## #SBATCH --qos=stud
# =============================================================================

# ── Environment setup ────────────────────────────────────────────────────────
module load miniconda3
module load cuda/12.4
source ~/.bashrc
conda activate transfer_env

# ── Paths — EDIT THIS ────────────────────────────────────────────────────────
# Where you uploaded the full iWildCam dataset (the folder containing
# metadata.csv and train/).  Example locations:
#   ~/HPC_EXPORT/data/iwildcam_v2.0   (if you put data inside HPC_EXPORT)
#   ~/data/iwildcam_v2.0               (if you uploaded data separately)
DATA_DIR=~/transferlearning/data

# ── Run ──────────────────────────────────────────────────────────────────────
cd ~/transferlearning

echo "=== Job $SLURM_JOB_ID started on $(hostname) at $(date) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== Python: $(python --version) ==="

python run_experiment.py \
    --data_dir    "$DATA_DIR" \
    --max_epochs  100 \
    --max_time_min 45

echo "=== Job finished at $(date) ==="
