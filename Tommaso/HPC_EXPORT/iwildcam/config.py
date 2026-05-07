import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_BASE_DIR)

CONFIG = {
    # --- data ---
    "data_dir":         os.path.join(_ROOT_DIR, "data"),
    "dataset":          "iwildcam",
    "dataset_mode":     "full",              # "mini" | "full"  ← switch here
    "num_classes":      182,                 # max(y)+1; same for mini and full
    "image_size":       224,

    # --- model ---
    "backbone":         "resnet50",

    # F/T/S partition over (stem, stage1, stage2, stage3, stage4, head).
    # Head must always be 'S'. Default: full fine-tune (T5S1).
    # Use models.enumerate_configs() to get all 21 valid monotonic configs.
    "partition":        ("T", "T", "T", "T", "T", "S"),

    # --- training ---
    "batch_size":       32,
    "epochs":           10,
    "seed":             42,
    "num_workers":      4,
    "device":           "cuda",           # falls back to cpu in train.py

    # --- optimizer ---
    "lr":               1e-4,             # backbone trainable blocks (T and S)
    "lr_head":          1e-3,             # head (always S)
    "weight_decay":     1e-3,
    "scheduler":        "cosine",         # cosine | step | none
    "label_smoothing":  0.1,              # 0.0 disables smoothing
    "head_dropout":     0.3,              # 0.0 disables dropout

    # --- class imbalance ---
    "use_weighted_sampler": True,

    # --- logging ---
    "save_dir":         os.path.join(_BASE_DIR, "checkpoints"),
    "log_dir":          os.path.join(_BASE_DIR, "results"),
    "use_wandb":        False,
    "project_name":     "transfer-learning-ood",
}
