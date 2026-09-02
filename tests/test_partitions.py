"""
Tests for the F/T/S partition algebra.

Pure logic -- no Torch, no dataset, no GPU. These run in milliseconds and are the
first thing to check after touching anything in ``partitions.py``.
"""

from __future__ import annotations

import pytest

from structured_transfer.partitions import (
    RESNET50_BLOCKS,
    WRN2810_BLOCKS,
    backbone_of,
    config_label,
    depths,
    enumerate_configs,
    parse_label,
    validate_partition,
)


class TestEnumeration:
    """The configuration space is (L+1)(L+2)/2 -- 21 and 15 in the two studies."""

    def test_resnet50_yields_21(self):
        assert len(enumerate_configs(RESNET50_BLOCKS)) == 21

    def test_wrn_yields_15(self):
        assert len(enumerate_configs(WRN2810_BLOCKS)) == 15

    def test_all_configs_are_unique(self):
        configs = enumerate_configs(RESNET50_BLOCKS)
        assert len(set(configs)) == len(configs)

    def test_all_configs_validate(self):
        for blocks in (RESNET50_BLOCKS, WRN2810_BLOCKS):
            for partition in enumerate_configs(blocks):
                validate_partition(partition, blocks)

    def test_matches_paper_appendix_b1(self):
        """The 15 WRN partitions listed in Appendix B.1, by backbone letters."""
        expected = {
            "SSSS",
            "FSSS", "TSSS",
            "FFSS", "FTSS", "TTSS",
            "FFFS", "FFTS", "FTTS", "TTTS",
            "FFFF", "FFFT", "FFTT", "FTTT", "TTTT",
        }
        got = {
            "".join(backbone_of(p)) for p in enumerate_configs(WRN2810_BLOCKS)
        }
        assert got == expected

    def test_matches_paper_appendix_b2(self):
        """The 21 iWildCam labels listed in Appendix B.2."""
        expected = {
            "S6",
            "F1S5", "T1S5",
            "F2S4", "F1T1S4", "T2S4",
            "F3S3", "F2T1S3", "F1T2S3", "T3S3",
            "F4S2", "F3T1S2", "F2T2S2", "F1T3S2", "T4S2",
            "F5S1", "F4T1S1", "F3T2S1", "F2T3S1", "F1T4S1", "T5S1",
        }
        got = {config_label(p) for p in enumerate_configs(RESNET50_BLOCKS)}
        assert got == expected


class TestLabels:
    @pytest.mark.parametrize("partition,expected", [
        (("F", "F", "F", "F", "F", "S"), "F5S1"),
        (("F", "F", "T", "T", "S", "S"), "F2T2S2"),
        (("S", "S", "S", "S", "S", "S"), "S6"),
        (("T", "T", "T", "T", "T", "S"), "T5S1"),
        (("F", "T", "T", "T", "T", "S"), "F1T4S1"),
    ])
    def test_config_label(self, partition, expected):
        assert config_label(partition) == expected

    def test_zero_count_modes_are_omitted(self):
        """T5S1, never F0T5S1 -- the archived filenames depend on this."""
        assert config_label(("T",) * 5 + ("S",)) == "T5S1"

    def test_round_trip_resnet(self):
        for partition in enumerate_configs(RESNET50_BLOCKS):
            assert parse_label(config_label(partition)) == partition

    def test_round_trip_wrn(self):
        for partition in enumerate_configs(WRN2810_BLOCKS):
            label = config_label(partition)
            assert parse_label(label, WRN2810_BLOCKS) == partition

    def test_parse_is_case_insensitive(self):
        assert parse_label("f2t3s1") == parse_label("F2T3S1")

    @pytest.mark.parametrize("bad", ["", "XYZ", "F2X1S3", "S1T5", "T5S1S1"])
    def test_malformed_labels_rejected(self, bad):
        with pytest.raises(ValueError):
            parse_label(bad)

    def test_wrong_block_count_rejected(self):
        """T5S1 is a 6-block label; asking for it on the 5-block WRN must fail."""
        with pytest.raises(ValueError, match="blocks"):
            parse_label("T5S1", WRN2810_BLOCKS)


class TestValidation:
    def test_head_must_be_scratch(self):
        # The target label space differs from the source, so a copied classifier
        # is meaningless. This is the one non-negotiable constraint.
        with pytest.raises(ValueError, match="Head"):
            validate_partition(("T", "T", "T", "T", "T", "T"), RESNET50_BLOCKS)

    def test_non_monotonic_rejected(self):
        # S below T: a reinitialized block would feed copied blocks a feature
        # distribution they were never trained on.
        with pytest.raises(ValueError, match="Non-monotonic"):
            validate_partition(("S", "T", "T", "T", "T", "S"), RESNET50_BLOCKS)

    def test_frozen_after_tuned_rejected(self):
        with pytest.raises(ValueError, match="Non-monotonic"):
            validate_partition(("T", "F", "T", "T", "T", "S"), RESNET50_BLOCKS)

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError, match="elements"):
            validate_partition(("F", "T", "S"), RESNET50_BLOCKS)

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="invalid mode"):
            validate_partition(("F", "F", "X", "T", "T", "S"), RESNET50_BLOCKS)


class TestDepths:
    def test_depths_exclude_the_head(self):
        # F5S1's only S is the head, so backbone scratch depth is 0.
        assert depths(parse_label("F5S1")) == {
            "frozen_depth": 5, "tuned_depth": 0, "scratch_depth": 0, "scratch_start": 5,
        }

    def test_all_scratch(self):
        assert depths(parse_label("S6")) == {
            "frozen_depth": 0, "tuned_depth": 0, "scratch_depth": 5, "scratch_start": 0,
        }

    def test_depths_sum_to_backbone_size(self):
        for partition in enumerate_configs(RESNET50_BLOCKS):
            d = depths(partition)
            assert d["frozen_depth"] + d["tuned_depth"] + d["scratch_depth"] == 5
            assert d["scratch_start"] == d["frozen_depth"] + d["tuned_depth"]
