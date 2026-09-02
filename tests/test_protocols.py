"""
Tests for the experiment protocols the runner scripts enumerate.

The job counts are the check that matters here. The paper's Experiment 1 is
"2 + 14x4 = 58 models"; if the protocol builder produces a different number,
something is wrong with how it identifies the baseline or the configuration
space, and a sweep would silently run the wrong set of jobs.

Imports the scripts by path, since ``scripts/`` is a CLI directory rather than
an installed package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load_script(name: str):
    """Import a script from scripts/ as a module, skipping if Torch is absent."""
    pytest.importorskip("torch")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTinyImageNetProtocol:
    """Experiment 1: 2 source networks + 14 partitions x 4 directions = 58."""

    @pytest.fixture(scope="class")
    def runner(self):
        return _load_script("tinyimagenet_run")

    def test_protocol_has_58_jobs(self, runner):
        assert len(runner.build_protocol()) == 58

    def test_baseline_appears_exactly_twice(self, runner):
        """
        SSSS copies nothing from a source, so A->B and B->A would retrain the
        identical model. It must appear once per target subset, not four times.
        """
        jobs = runner.build_protocol()
        baseline = [j for j in jobs if j[0] == "SSSS"]
        assert baseline == [("SSSS", "AtoA"), ("SSSS", "BtoB")]

    def test_labels_use_the_4_letter_backbone_form(self, runner):
        """
        Tables 4 and 5 name configurations by backbone letters (FFTS), not the
        counted form (F2T1S2). Mixing the two silently breaks the baseline check:
        config_label('SSSS' + head) is 'S5', which never equals 'SSSS', so the
        baseline would be scheduled four times and the protocol would be 62 jobs.
        """
        for label, _ in runner.build_protocol():
            assert len(label) == 4, f"{label!r} is not a 4-letter backbone label"
            assert set(label) <= {"F", "T", "S"}
            assert not any(ch.isdigit() for ch in label)

    def test_every_non_baseline_config_gets_all_four_directions(self, runner):
        from collections import Counter

        counts = Counter(label for label, _ in runner.build_protocol())
        assert counts["SSSS"] == 2
        for label, count in counts.items():
            if label != "SSSS":
                assert count == 4, f"{label} appears {count} times, expected 4"

    def test_14_source_dependent_configurations(self, runner):
        labels = {label for label, _ in runner.build_protocol()}
        assert len(labels - {"SSSS"}) == 14

    def test_labels_parse_back_to_valid_partitions(self, runner):
        from structured_transfer.partitions import WRN2810_BLOCKS, validate_partition

        for label, _ in runner.build_protocol():
            partition = runner._partition_from_label(label)
            validate_partition(partition, WRN2810_BLOCKS)
            assert runner.backbone_label(label) == label

    def test_counted_labels_are_also_accepted(self, runner):
        """
        A user may reasonably type either form on the command line.

        FFTS is conv1=F, group1=F, group2=T, group3=S, plus the scratch head --
        so counted it is F2 T1 S2, not F2T1S1.
        """
        assert runner._partition_from_label("FFTS") == ("F", "F", "T", "S", "S")
        assert runner._partition_from_label("FFTS") == \
               runner._partition_from_label("F2T1S2")

    def test_bad_label_length_is_rejected(self, runner):
        with pytest.raises(SystemExit, match="letters"):
            runner._partition_from_label("FFT")


class TestIWildCamProtocol:
    """Experiment 2: 21 partitions, index-addressable for SLURM arrays."""

    @pytest.fixture(scope="class")
    def runner(self):
        return _load_script("iwildcam_run")

    def test_smoke_partitions_all_parse(self, runner):
        from structured_transfer.partitions import validate_partition

        for label in runner.SMOKE_PARTITIONS:
            validate_partition(runner.parse_label(label))

    def test_smoke_spans_the_design_space(self, runner):
        """The smoke set should cover fully frozen, mixed, full FT and scratch."""
        assert "F5S1" in runner.SMOKE_PARTITIONS   # fully frozen
        assert "T5S1" in runner.SMOKE_PARTITIONS   # full fine-tune
        assert "S6" in runner.SMOKE_PARTITIONS     # no pretraining

    def test_index_selection_covers_all_21(self, runner):
        import argparse

        seen = set()
        for i in range(21):
            args = argparse.Namespace(
                all=False, smoke=False, index=i, partition=None,
            )
            (partition,) = runner.resolve_partitions(args)
            seen.add(runner.config_label(partition))
        assert len(seen) == 21

    def test_out_of_range_index_is_rejected(self, runner):
        import argparse

        args = argparse.Namespace(all=False, smoke=False, index=21, partition=None)
        with pytest.raises(SystemExit, match=r"--index"):
            runner.resolve_partitions(args)

    def test_all_flag_yields_21(self, runner):
        import argparse

        args = argparse.Namespace(all=True, smoke=False, index=None, partition=None)
        assert len(runner.resolve_partitions(args)) == 21
