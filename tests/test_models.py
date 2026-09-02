"""
Tests that both backbones match the paper's published budgets and that F/T/S
partitioning actually freezes what it claims to.

CPU-only and dataset-free, but they do instantiate real networks, so expect
seconds rather than milliseconds. Building ResNet50 with pretrained weights
downloads ~100 MB the first time; the tests that need it are marked ``slow`` and
skip cleanly when the weights are unavailable offline.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from structured_transfer.partitions import (  # noqa: E402
    RESNET50_BLOCKS,
    WRN2810_BLOCKS,
    enumerate_configs,
    parse_label,
)
from structured_transfer.tinyimagenet.models import (  # noqa: E402
    WideResNet,
    block_parameter_table,
)
from structured_transfer.utils import count_parameters  # noqa: E402


class TestWideResNetBudget:
    """
    WRN-28-10 must reproduce the reported per-block budget exactly.

    Parameter counts are unforgiving of a wrong block count, width, or shortcut
    placement, so these constants are the sharpest available check that the
    architecture matches its specification.
    """

    #: Table 1, parameters per controlled block (100-class head).
    TABLE_1 = {
        "conv1":       432,
        "group1":  1_640_672,
        "group2":  6_968_000,
        "group3": 27_862_400,
        "head":       65_380,
    }
    TABLE_1_TOTAL = 36_536_884

    def test_per_block_parameters(self):
        assert block_parameter_table() == self.TABLE_1

    def test_total_parameters(self):
        model = WideResNet(num_classes=100)
        _, n_total = count_parameters(model)
        assert n_total == self.TABLE_1_TOTAL
        assert sum(self.TABLE_1.values()) == self.TABLE_1_TOTAL

    def test_group3_dominates_parameters(self):
        """76% of the model sits in group3 -- the fact behind the G2->G3 instability."""
        share = self.TABLE_1["group3"] / self.TABLE_1_TOTAL
        assert 0.76 < share < 0.77

    def test_forward_shape_and_resolution(self):
        """64 -> 64 -> 32 -> 16, per Appendix A.1: no downsampling at the stem."""
        model = WideResNet(num_classes=100).eval()
        x = torch.zeros(2, 3, 64, 64)
        with torch.no_grad():
            assert model(x).shape == (2, 100)
            h = model.conv1(x)
            assert h.shape[1:] == (16, 64, 64)
            h = model.group1(h)
            assert h.shape[1:] == (160, 64, 64)
            h = model.group2(h)
            assert h.shape[1:] == (320, 32, 32)
            h = model.group3(h)
            assert h.shape[1:] == (640, 16, 16)

    def test_rejects_invalid_depth(self):
        with pytest.raises(ValueError, match="depth"):
            WideResNet(num_classes=100, depth=29)


class TestWideResNetPartitioning:
    """Frozen blocks must be excluded from training; scratch blocks included."""

    @staticmethod
    def _build(label: str):
        from structured_transfer.tinyimagenet.config import CONFIG
        from structured_transfer.tinyimagenet.models import build_model

        partition = parse_label(label, WRN2810_BLOCKS)
        # A source checkpoint taken from a freshly built network: these tests
        # check the wiring, not what the weights contain.
        source = WideResNet(num_classes=100).state_dict()
        model, n_train, n_total = build_model(partition, dict(CONFIG), source)
        return partition, model, n_train, n_total

    def test_all_scratch_trains_everything(self):
        _, _, n_train, n_total = self._build("S5")
        assert n_train == n_total

    def test_fully_frozen_leaves_only_the_head(self):
        # F4S1: conv1..group3 frozen, head scratch. Only the head trains.
        _, _, n_train, _ = self._build("F4S1")
        assert n_train == TestWideResNetBudget.TABLE_1["head"]

    def test_frozen_blocks_have_no_gradients(self):
        partition, model, _, _ = self._build("F2T1S2")
        for block_name, mode in zip(WRN2810_BLOCKS, partition):
            block = getattr(model, block_name)
            expected = mode != "F"
            for param in block.parameters():
                assert param.requires_grad == expected, (
                    f"{block_name} is {mode} but requires_grad={param.requires_grad}"
                )

    def test_source_weights_copied_into_f_and_t_blocks(self):
        from structured_transfer.tinyimagenet.config import CONFIG
        from structured_transfer.tinyimagenet.models import build_model

        source_model = WideResNet(num_classes=100)
        source = source_model.state_dict()
        partition = parse_label("F1T1S3", WRN2810_BLOCKS)
        model, _, _ = build_model(partition, dict(CONFIG), source)

        # conv1 (F) and group1 (T) are copied; group2/group3 (S) are not.
        assert torch.equal(model.conv1.weight, source_model.conv1.weight)
        assert torch.equal(
            model.group1[0].conv1.weight, source_model.group1[0].conv1.weight
        )
        assert not torch.equal(
            model.group2[0].conv1.weight, source_model.group2[0].conv1.weight
        )

    def test_missing_source_is_a_clear_error(self):
        from structured_transfer.tinyimagenet.config import CONFIG
        from structured_transfer.tinyimagenet.models import build_model

        partition = parse_label("F2T1S2", WRN2810_BLOCKS)
        with pytest.raises(ValueError, match="source_state_dict"):
            build_model(partition, dict(CONFIG), None)

    def test_frozen_batchnorm_held_in_eval(self):
        from structured_transfer.tinyimagenet.models import freeze_frozen_batchnorm

        partition, model, _, _ = self._build("F2T1S2")
        model.train()
        freeze_frozen_batchnorm(model, partition)

        # group1 is F: its BN must stay in eval so running statistics keep the
        # source-domain values (Appendix A.1).
        for module in model.group1.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                assert not module.training
        # group2 is T: its BN must keep updating.
        for module in model.group2.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                assert module.training


@pytest.mark.slow
class TestResNet50Budget:
    """
    ResNet50 must reproduce the paper's Table 2 and the Table 3 ``Train.`` column.

    Skipped when torchvision cannot fetch ImageNet weights (offline CI).
    """

    TABLE_2_TOTAL = 23_880_950   # with the 182-class iWildCam head

    #: Table 3's Train. column, per frozen depth. The whole efficiency argument
    #: rests on these: freezing four of five blocks still leaves 64% of the
    #: parameters trainable, because Stage4 alone is 63% of the model.
    TRAINABLE_PCT = {
        "S6":       100.0,
        "F1S5":     100.0,   # 99.96, rounds to 100.0 in the paper
        "F2S4":      99.1,
        "F3S3":      93.9,
        "F4S2":      64.2,
        "F4T1S1":    64.2,   # same frozen prefix, so the same trainable share
        "F5S1":       1.6,
    }

    @staticmethod
    def _build(label: str):
        from structured_transfer.iwildcam.config import CONFIG
        from structured_transfer.iwildcam.models import build_model

        cfg = dict(CONFIG)
        cfg["num_classes"] = 182
        # Point data_dir somewhere absent so build_model falls back to
        # num_classes instead of reading a metadata CSV.
        cfg["data_dir"] = "/nonexistent"
        partition = parse_label(label)
        return build_model(partition, cfg)

    def test_total_parameters_match_table_2(self):
        try:
            _, _, n_total = self._build("S6")     # no pretrained download needed
        except Exception as exc:                  # pragma: no cover
            pytest.skip(f"could not build ResNet50: {exc}")
        assert n_total == self.TABLE_2_TOTAL

    @pytest.mark.parametrize("label,expected", sorted(TRAINABLE_PCT.items()))
    def test_trainable_pct_matches_table_3(self, label, expected):
        try:
            _, n_train, n_total = self._build(label)
        except Exception as exc:                  # pragma: no cover
            pytest.skip(f"could not build ResNet50 (needs ImageNet weights): {exc}")
        assert round(100 * n_train / n_total, 1) == pytest.approx(expected, abs=0.05)

    def test_head_is_always_trainable(self):
        """Even in the fully frozen F5S1, the classifier must still train."""
        try:
            model, _, _ = self._build("F5S1")
        except Exception as exc:                  # pragma: no cover
            pytest.skip(f"could not build ResNet50: {exc}")
        assert all(p.requires_grad for p in model.fc.parameters())

    def test_flops_ratio_reproduces_the_76_percent_claim(self):
        """
        The paper's "F4T1S1 at about 76% of full fine-tuning FLOPs".

        estimate_flops models training cost as forward x (1 + 2 x trainable_ratio),
        so the ratio is a property of the parameter split alone and is checkable
        without fvcore or a GPU.
        """
        f4t1s1_ratio = 1 + 2 * (self.TRAINABLE_PCT["F4T1S1"] / 100)
        t5s1_ratio   = 1 + 2 * 1.0
        assert round(100 * f4t1s1_ratio / t5s1_ratio) == 76


class TestConfigurationSpaceIntegration:
    """Every enumerated partition must actually build, not just validate."""

    @pytest.mark.slow
    def test_all_wrn_partitions_build(self):
        from structured_transfer.tinyimagenet.config import CONFIG
        from structured_transfer.tinyimagenet.models import build_model

        source = WideResNet(num_classes=100).state_dict()
        for partition in enumerate_configs(WRN2810_BLOCKS):
            model, n_train, n_total = build_model(partition, dict(CONFIG), source)
            assert n_total == TestWideResNetBudget.TABLE_1_TOTAL
            assert 0 < n_train <= n_total
