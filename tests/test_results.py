"""
Tests over the committed result artifacts.

These are the integrity checks on the data itself: the 21 archived runs must
still parse, the derived leaderboard must still agree with the paper's Table 3
on the 20 configurations that matched when it was generated, and the transcribed
tables must be internally consistent.

Needs only pandas, no Torch.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pandas")

from structured_transfer.analysis.load import (  # noqa: E402
    RESULTS_DIR,
    load_leaderboard,
    load_run_curves,
    load_runs,
    load_tinyimagenet_tables,
)
from structured_transfer.partitions import RESNET50_BLOCKS, config_label, enumerate_configs  # noqa: E402

RUNS_DIR = RESULTS_DIR / "iwildcam" / "runs"

#: The single configuration where the archived run and the reference table
#: diverge. Its run terminated at 19.6 epochs on the wall-clock ceiling, whereas
#: the reference row reflects a longer run.
KNOWN_DIVERGENCE = "T5S1"


@pytest.fixture(scope="module")
def runs():
    if not RUNS_DIR.exists():
        pytest.skip(f"no run artifacts at {RUNS_DIR}")
    return load_runs(RUNS_DIR)


class TestRunArtifacts:
    def test_all_21_configurations_present(self, runs):
        expected = {config_label(p) for p in enumerate_configs(RESNET50_BLOCKS)}
        assert set(runs.index) == expected

    def test_every_run_parses_with_the_expected_schema(self):
        required = {
            "label", "config", "n_trainable", "n_total",
            "inference_flops", "training_flops_per_sample",
            "epochs_completed", "train_elapsed_s", "eval_log", "final_results",
        }
        for path in sorted(RUNS_DIR.glob("timing_*.json")):
            with open(path, encoding="utf-8") as handle:
                run = json.load(handle)
            missing = required - set(run)
            assert not missing, f"{path.name} is missing {missing}"

    def test_all_runs_share_one_protocol(self, runs):
        """
        Comparability depends on this: same dataset, batch size, LRs and
        regularization across all 21. A stray override would silently invalidate
        the leaderboard.
        """
        protocol_keys = ["dataset_mode", "batch_size", "lr", "lr_head",
                         "weight_decay", "label_smoothing", "head_dropout",
                         "use_weighted_sampler", "seed"]
        seen: dict[str, set] = {k: set() for k in protocol_keys}
        for path in sorted(RUNS_DIR.glob("timing_*.json")):
            with open(path, encoding="utf-8") as handle:
                cfg = json.load(handle)["config"]
            for key in protocol_keys:
                seen[key].add(cfg.get(key))
        for key, values in seen.items():
            assert len(values) == 1, f"{key} varies across runs: {values}"

    def test_total_parameters_are_constant(self, runs):
        """Partitioning changes what trains, never the architecture."""
        assert runs["total_params"].nunique() == 1
        assert runs["total_params"].iloc[0] == 23_880_950

    def test_inference_flops_are_constant(self, runs):
        """Freezing changes the backward pass only, so forward cost is fixed."""
        assert runs["inference_gflops"].round(2).nunique() == 1

    def test_curves_cover_every_run(self, runs):
        curves = load_run_curves(RUNS_DIR)
        assert set(curves["config"]) == set(runs.index)
        assert (curves.groupby("config")["epoch"].count() > 0).all()


class TestLeaderboardAgreement:
    def test_derived_matches_paper_on_20_of_21(self, runs):
        paper = load_leaderboard(prefer="paper")
        metrics = ["id_acc", "id_bal_acc", "id_f1",
                   "ood_acc", "ood_bal_acc", "ood_f1"]

        mismatched = {
            config for config in paper.index
            if any(
                abs(runs.loc[config, m] - paper.loc[config, m]) > 0.051
                for m in metrics
            )
        }
        assert mismatched == {KNOWN_DIVERGENCE}, (
            f"Expected only {KNOWN_DIVERGENCE} to differ from the paper, "
            f"got {sorted(mismatched)}."
        )

    def test_divergent_run_was_truncated(self, runs):
        """
        Document the cause, not just the symptom: T5S1 differs because its run
        was cut short, and it is the slowest configuration in the sweep.
        """
        truncated = runs.loc[KNOWN_DIVERGENCE]
        assert truncated["epochs_completed"] < 25
        assert truncated["secs_per_epoch"] == runs["secs_per_epoch"].max()

    def test_committed_csv_matches_the_derived_table(self, runs):
        import pandas as pd

        path = RESULTS_DIR / "iwildcam" / "leaderboard.csv"
        if not path.exists():
            pytest.skip("leaderboard.csv not committed")
        committed = pd.read_csv(path).set_index("config")
        assert set(committed.index) == set(runs.index)
        for metric in ["id_acc", "ood_acc", "ood_bal_acc", "trainable_pct"]:
            for config in runs.index:
                assert committed.loc[config, metric] == pytest.approx(
                    runs.loc[config, metric], abs=0.051
                ), f"{config}.{metric} disagrees with the run artifacts"


class TestPaperTables:
    def test_iwildcam_table_has_21_rows(self):
        paper = load_leaderboard(prefer="paper")
        expected = {config_label(p) for p in enumerate_configs(RESNET50_BLOCKS)}
        assert set(paper.index) == expected

    def test_depth_columns_are_self_consistent(self):
        paper = load_leaderboard(prefer="paper")
        for config, row in paper.iterrows():
            total = row["frozen_depth"] + row["tuned_depth"] + row["scratch_depth"]
            assert total == 5, f"{config}: depths sum to {total}, expected 5"
            assert row["scratch_start"] == row["frozen_depth"] + row["tuned_depth"]

    def test_tinyimagenet_tables_have_15_rows_each(self):
        selfer, transfer = load_tinyimagenet_tables()
        assert len(selfer) == 15
        assert len(transfer) == 15
        assert set(selfer.index) == set(transfer.index)

    def test_deltas_are_consistent_with_the_baseline(self):
        """delta_clean must equal clean_acc minus the SSSS baseline."""
        for table in load_tinyimagenet_tables():
            baseline = table.loc["SSSS", "clean_acc"]
            for config, row in table.iterrows():
                assert row["delta_clean"] == pytest.approx(
                    row["clean_acc"] - baseline, abs=0.051
                ), f"{config}: delta_clean is inconsistent with clean_acc"

    def test_headline_findings_hold(self):
        """
        Guard the three claims the write-up leads with. If a transcription slips,
        these fail rather than silently changing the story.
        """
        paper = load_leaderboard(prefer="paper")
        selfer, transfer = load_tinyimagenet_tables()

        # 1. A fully frozen ImageNet backbone is worse than training from scratch.
        assert paper.loc["F5S1", "ood_acc"] < paper.loc["S6", "ood_acc"]

        # 2. Full fine-tuning gives the best OOD balanced accuracy.
        assert paper["ood_bal_acc"].idxmax() == "T5S1"

        # 3. Cross-domain frozen FFFF collapses; same-domain FFFF does not.
        assert transfer.loc["FFFF", "clean_acc"] < 55
        assert selfer.loc["FFFF", "clean_acc"] > 73
