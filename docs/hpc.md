# Running on a cluster

The iWildCam sweep is 21 GPU jobs of about three hours each; the TinyImageNet
protocol is 58 jobs of WRN-28-10 for 100 epochs. Both are array jobs.

## Cost, measured

From the archived runs (`results/iwildcam/runs/`), on an **NVIDIA A100-SXM-64GB**:

| | Value |
|---|---|
| Configurations | 21 |
| Total GPU time | **65 hours** |
| Per-run training time | 3.07 – 3.23 h (the wall-clock ceiling, so nearly constant) |
| Seconds per epoch | 251 (median 282) – 592 |
| Epochs completed | 19.6 – 44.3 (median 39.5) |

Per-run *time* is almost constant because the budget is wall clock; what varies
is how many epochs fit inside it.

TinyImageNet has not been run with this code, so no measured figures exist. As a
planning estimate, WRN-28-10 is ~36.5 M parameters and ~23.8 GFLOPs forward at
64×64 against ResNet50's ~4.1 GFLOPs at 224×224, over 50,000 training images —
budget several hours per run and treat the 58-job array as an overnight-plus job.

## Environment

```bash
module load miniconda3
module load cuda/12.4
conda create -n transfer_env python=3.11
conda activate transfer_env

# Install a torch build matching the cluster's CUDA first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

cd $HOME/Transfer_Learning_AI_project
pip install -e ".[dev,flops]"
```

`fvcore` is worth installing. Without it `estimate_flops()` falls back to a
published constant and will not reproduce the paper's 4.11 GFLOPs figure — the
archived runs all have it.

## iWildCam sweep

```bash
# 1. Stage the dataset (metadata + images) somewhere with real bandwidth
python scripts/iwildcam_download_metadata.py --data_dir $SCRATCH/tl/data

# 2. Check the index -> configuration mapping
python scripts/iwildcam_run.py --list

# 3. Submit
DATA_DIR=$SCRATCH/tl/data sbatch slurm/iwildcam_sweep.sbatch

# 4. Collect
squeue -u $USER
python scripts/make_leaderboard.py
python scripts/make_figures.py --source runs
```

Each array task writes one `results/iwildcam/runs/timing_<LABEL>_full.json`, so
tasks are independent and a failed one can be resubmitted alone with
`sbatch --array=N`.

## TinyImageNet sweep

**Task order matters.** Array tasks 0 and 1 train the two source networks
(`baseA`, `baseB`); every other task loads one of them and exits with a clear
error if it is missing.

```bash
# 1. Prepare the class split and the corrupted validation sets
python scripts/tinyimagenet_prepare.py --stage all --data_dir $SCRATCH/tl/data

# 2. Source networks first, then everything else
BASE=$(sbatch --parsable --array=0-1 slurm/tinyimagenet_sweep.sbatch)
sbatch --dependency=afterok:$BASE --array=2-57 slurm/tinyimagenet_sweep.sbatch
```

## Resource sizing

| Setting | iWildCam | TinyImageNet | Why |
|---|---|---|---|
| `--cpus-per-task` | 4 | 4 | Must match `num_workers=4`; fewer starves the GPU on 224×224 JPEG decode |
| `--mem` | 32G | 24G | iWildCam holds a 203k-row metadata frame plus decode buffers |
| `--gres` | `gpu:1` | `gpu:1` | Single-GPU by design; no distributed training |
| `--time` | 4h | 6h | Training ceiling plus final evaluation plus margin |

Data loading, not the GPU, is usually the bottleneck. Stage the dataset on the
cluster's fast scratch filesystem, never on a network home directory.

## Interpreting the time budget

`EpochTimeBudgetTracker` stops on epochs or wall clock, whichever comes first.
Equal wall clock is what makes 21 jobs schedulable on a shared machine, but it
implies that a slower run completes fewer epochs, which is a confound worth
stating explicitly.

One configuration was affected. T5S1 ran at 592 s/epoch and completed 19.6
epochs, against 35–44 for the remainder. This is not explained by the partition
being more expensive: T4S2 optimizes the same 100% of parameters and ran at
281 s/epoch, and the slowest other configuration reached 321 s/epoch. A twofold
slowdown with an unchanged computational graph is most consistent with contention
or an I/O stall on the shared node during that job. See
[`results/iwildcam/README.md`](../results/iwildcam/README.md).

For a new sweep, a fixed epoch budget is preferable: pass a generous
`--max_time_min` and let `--max_epochs` bind instead. This removes the confound
at the cost of longer and less predictable queue times. The TinyImageNet protocol
is epoch-fixed for this reason.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Source network not found` | TinyImageNet tasks 0–1 have not finished. Run them first. |
| `No such file: metadata.csv` | `DATA_DIR` does not point at the parent of `iwildcam_v2.0/`. |
| Very low GPU utilisation | Data loading is starving the GPU: raise `--cpus-per-task` and `num_workers` together, and confirm the data is on fast scratch. |
| `CUDA out of memory` | Lower `batch_size` in `src/structured_transfer/iwildcam/config.py`. Note this changes the protocol, so results stop being comparable to the archived runs. |
| Wildly varying seconds-per-epoch | Node contention, as with T5S1 above. Compare `secs_per_epoch` across runs before trusting a leaderboard built from them. |
