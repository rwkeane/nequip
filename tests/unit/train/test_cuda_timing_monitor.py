import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nequip.train.callbacks.cuda_timing_monitor import CUDATimingMonitor


class TestCUDATimingMonitor:
    def test_write_json_schema(self, tmp_path):
        monitor = CUDATimingMonitor(
            output_path=str(tmp_path / "timing.json"), enabled=False
        )
        monitor._records = [
            {
                "epoch": 0,
                "train": {"gpu_ms": 100.5, "num_batches": 10},
                "val": {"gpu_ms": 12.3, "num_batches": 2},
            }
        ]
        trainer = SimpleNamespace(is_global_zero=True, logger=None)

        monitor._write_json(trainer)

        with open(tmp_path / "timing.json", encoding="utf-8") as f:
            payload = json.load(f)

        assert payload["epochs"] == monitor._records
        assert payload["summary"] == {
            "total_train_gpu_ms": 100.5,
            "total_val_gpu_ms": 12.3,
            "num_epochs": 1,
        }

    def test_finalize_epoch_record_only_on_global_zero(self, tmp_path):
        monitor = CUDATimingMonitor(
            output_path=str(tmp_path / "timing.json"), enabled=False
        )
        monitor._current = {
            "epoch": 1,
            "train": {"gpu_ms": 50.0, "num_batches": 5},
        }
        trainer = SimpleNamespace(is_global_zero=False, logger=None)

        monitor._finalize_epoch_record(trainer)

        assert monitor._records == [
            {"epoch": 1, "train": {"gpu_ms": 50.0, "num_batches": 5}}
        ]
        assert not (tmp_path / "timing.json").exists()

    @pytest.mark.parametrize(
        ("check_val_every_n_epoch", "current_epoch", "expected"),
        [
            (1, 0, True),
            (2, 0, False),
            (2, 1, True),
            (0, 0, False),
        ],
    )
    def test_validation_runs_this_epoch(
        self, check_val_every_n_epoch, current_epoch, expected
    ):
        monitor = CUDATimingMonitor(enabled=False)
        trainer = MagicMock()
        trainer.check_val_every_n_epoch = check_val_every_n_epoch
        trainer.current_epoch = current_epoch

        assert monitor._validation_runs_this_epoch(trainer) is expected

    def test_batch_count_handles_lists(self):
        monitor = CUDATimingMonitor(enabled=False)
        trainer = SimpleNamespace(num_training_batches=[3, 4], num_val_batches=[2, 1])

        assert monitor._batch_count(trainer, "num_training_batches") == 7
        assert monitor._batch_count(trainer, "num_val_batches") == 3
