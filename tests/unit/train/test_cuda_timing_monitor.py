import json
from types import SimpleNamespace

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

    def test_phase_timings_merge_regardless_of_hook_order(self, tmp_path):
        monitor = CUDATimingMonitor(
            output_path=str(tmp_path / "timing.json"), enabled=False
        )
        # Lightning ends validation before calling on_train_epoch_end.
        monitor._store_timing(1, "val", 5.0, 2)
        monitor._store_timing(1, "train", 50.0, 5)
        assert monitor._records == [
            {
                "epoch": 1,
                "val": {"gpu_ms": 5.0, "num_batches": 2},
                "train": {"gpu_ms": 50.0, "num_batches": 5},
            }
        ]

    def test_store_timing_updates_existing_phase(self):
        monitor = CUDATimingMonitor(enabled=False)
        monitor._store_timing(0, "val", 1.0, 1)
        monitor._store_timing(0, "val", 2.0, 2)

        assert monitor._records == [
            {"epoch": 0, "val": {"gpu_ms": 2.0, "num_batches": 2}}
        ]

    def test_validation_start_ends_training_once(self, monkeypatch):
        monitor = CUDATimingMonitor(enabled=False)
        monitor.enabled = True
        calls = []
        monkeypatch.setattr(
            monitor, "_record_start", lambda phase: calls.append(("start", phase))
        )
        monkeypatch.setattr(
            monitor,
            "_record_end",
            lambda trainer, phase: calls.append(("end", phase)),
        )
        monkeypatch.setattr(monitor, "_write_if_global_zero", lambda trainer: None)
        trainer = SimpleNamespace(sanity_checking=False)

        monitor.on_train_epoch_start(trainer, None)
        monitor.on_validation_epoch_start(trainer, None)
        monitor.on_train_epoch_end(trainer, None)

        assert calls == [
            ("start", "train"),
            ("end", "train"),
            ("start", "val"),
        ]

    def test_batch_count_handles_lists(self):
        monitor = CUDATimingMonitor(enabled=False)
        trainer = SimpleNamespace(num_training_batches=[3, 4], num_val_batches=[2, 1])

        assert monitor._batch_count(trainer, "num_training_batches") == 7
        assert monitor._batch_count(trainer, "num_val_batches") == 3
