# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import json
import os
from pathlib import Path
from typing import Any, Optional

import lightning
import torch
from lightning.pytorch.callbacks import Callback

from nequip.utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class CUDATimingMonitor(Callback):
    """Measure train/val epoch GPU time with CUDA events and write ``timing.json``.

    CUDA events time work on the GPU timeline for each epoch. This excludes CPU-side
    dataloader overhead and is typically lower than wall-clock epoch time.

    Logged metrics (per phase: ``train`` / ``val``):

    - ``time/<phase>_epoch_gpu_ms`` – GPU milliseconds for the epoch

    Args:
        output_path (str | None): Path for the JSON timing file. Defaults to
            ``timing.json`` in Hydra's output directory when available.
        enabled (bool): Whether timing is active. Automatically disabled when CUDA
            is unavailable.
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        enabled: bool = True,
    ):
        self.output_path = output_path
        self.enabled = enabled and torch.cuda.is_available()
        self._records: list[dict[str, Any]] = []
        self._train_start: Optional[torch.cuda.Event] = None
        self._val_start: Optional[torch.cuda.Event] = None
        self._train_active = False

    def setup(self, trainer: "lightning.Trainer", pl_module: lightning.LightningModule, stage: str):
        if self.enabled:
            path = self._resolve_output_path(trainer)
            logger.info(f"CUDATimingMonitor enabled; writing timing to {path}")
        else:
            logger.info("CUDATimingMonitor disabled (CUDA not available).")

    def _resolve_output_path(self, trainer: "lightning.Trainer") -> Path:
        if self.output_path is not None:
            return Path(self.output_path)
        try:
            import hydra

            return (
                Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
                / "timing.json"
            )
        except Exception:
            if trainer.logger is not None and hasattr(trainer.logger, "log_dir"):
                return Path(trainer.logger.log_dir) / "timing.json"
            return Path("timing.json")

    @staticmethod
    def _batch_count(trainer: "lightning.Trainer", attr: str) -> int:
        count = getattr(trainer, attr, None)
        if count is None:
            return 0
        if isinstance(count, (list, tuple)):
            return int(sum(count))
        return int(count)

    def _record_start(self, phase: str) -> None:
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        if phase == "train":
            self._train_start = start
        else:
            self._val_start = start

    def _record_end(self, trainer: "lightning.Trainer", phase: str) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        end.synchronize()

        if phase == "train":
            gpu_ms = self._train_start.elapsed_time(end)
            num_batches = self._batch_count(trainer, "num_training_batches")
        else:
            gpu_ms = self._val_start.elapsed_time(end)
            num_batches = self._batch_count(trainer, "num_val_batches")

        self._store_timing(trainer.current_epoch, phase, gpu_ms, num_batches)

        trainer.lightning_module.log(
            f"time/{phase}_epoch_gpu_ms",
            gpu_ms,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )

        print(
            f"[CUDATimingMonitor] epoch={trainer.current_epoch} phase={phase} gpu_ms={gpu_ms:.1f}",
            flush=True,
        )

    def _get_epoch_record(self, epoch: int) -> dict[str, Any]:
        for record in self._records:
            if record["epoch"] == epoch:
                return record
        record: dict[str, Any] = {"epoch": epoch}
        self._records.append(record)
        return record

    def _store_timing(
        self, epoch: int, phase: str, gpu_ms: float, num_batches: int
    ) -> None:
        self._get_epoch_record(epoch)[phase] = {
            "gpu_ms": gpu_ms,
            "num_batches": num_batches,
        }

    def _write_if_global_zero(self, trainer: "lightning.Trainer") -> None:
        if trainer.is_global_zero:
            self._write_json(trainer)

    def _finish_train(self, trainer: "lightning.Trainer") -> None:
        if not self._train_active:
            return
        self._record_end(trainer, "train")
        self._train_active = False
        self._write_if_global_zero(trainer)

    def _compute_summary(self) -> dict[str, float | int]:
        return {
            "total_train_gpu_ms": sum(
                record.get("train", {}).get("gpu_ms", 0.0) for record in self._records
            ),
            "total_val_gpu_ms": sum(
                record.get("val", {}).get("gpu_ms", 0.0) for record in self._records
            ),
            "num_epochs": len(self._records),
        }

    def _write_json(self, trainer: "lightning.Trainer") -> None:
        path = self._resolve_output_path(trainer)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "epochs": self._records,
            "summary": self._compute_summary(),
        }

        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)

    def on_train_epoch_start(self, trainer, pl_module):
        if not self.enabled:
            return
        self._record_start("train")
        self._train_active = True

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.enabled:
            return
        # Lightning calls this after validation, so this is only a fallback for
        # epochs where validation did not run.
        self._finish_train(trainer)

    def on_validation_epoch_start(self, trainer, pl_module):
        if not self.enabled or trainer.sanity_checking:
            return
        # Lightning runs validation before on_train_epoch_end. End the training
        # interval here so validation GPU work is not counted as training.
        self._finish_train(trainer)
        self._record_start("val")

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.enabled or trainer.sanity_checking:
            return
        self._record_end(trainer, "val")
        self._write_if_global_zero(trainer)

    def on_fit_end(self, trainer, pl_module):
        if not self.enabled:
            return
        if trainer.is_global_zero:
            self._write_json(trainer)

    def on_validation_end(self, trainer, pl_module):
        if not self.enabled:
            return
        if trainer.is_global_zero:
            self._write_json(trainer)
