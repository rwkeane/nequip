# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import time
import torch
import lightning
from lightning.pytorch.callbacks import Callback


class GPUMemoryMonitor(Callback):
    """Log peak GPU memory and epoch wall-clock time at the end of each train/val epoch.

    Memory is reported in **mebibytes (MiB)**. At the start of every epoch the
    CUDA peak memory stats are reset, so `max_memory_allocated` reflects the
    true high-water mark *within that epoch only* — capturing the worst-case
    live memory (weights + optimizer states + activations) at any single point
    during the epoch.

    Logged metrics (per phase: ``train`` / ``val``):

    - ``memory/<phase>_peak_allocated_MiB`` – peak memory held by live tensors
      at any point during the epoch
    - ``memory/<phase>_peak_reserved_MiB``  – peak memory reserved by the caching
      allocator during the epoch
    - ``time/<phase>_epoch_s``              – wall-clock seconds for the epoch

    Args:
        devices (list[int] | None): CUDA device indices to monitor.  Defaults to
            all devices visible to the current process.
    """

    def __init__(self, devices: list = None):
        self._devices = devices
        self._train_start: float = 0.0
        self._val_start: float = 0.0

    def _get_devices(self):
        if self._devices is not None:
            return self._devices
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))

    def _reset(self):
        for dev in self._get_devices():
            torch.cuda.reset_peak_memory_stats(dev)

    def _log_and_print(self, trainer: "lightning.Trainer", prefix: str, elapsed_s: float):
        epoch = trainer.current_epoch
        lines = [f"[GPUMemoryMonitor] epoch={epoch} phase={prefix}  wall={elapsed_s:.1f}s"]
        for dev in self._get_devices():
            alloc = torch.cuda.max_memory_allocated(dev) / 2**20
            reserved = torch.cuda.max_memory_reserved(dev) / 2**20
            tag = f"cuda:{dev}/" if len(self._get_devices()) > 1 else ""

            trainer.lightning_module.log(
                f"memory/{tag}{prefix}_peak_allocated_MiB",
                alloc,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
            )
            trainer.lightning_module.log(
                f"memory/{tag}{prefix}_peak_reserved_MiB",
                reserved,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
            )
            lines.append(
                f"  {tag}peak_allocated={alloc:.1f} MiB  peak_reserved={reserved:.1f} MiB"
            )

        trainer.lightning_module.log(
            f"time/{prefix}_epoch_s",
            elapsed_s,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )

        print("\n" + "\n".join(lines), flush=True)

    # --- train ---
    def on_train_epoch_start(self, trainer, pl_module):
        self._reset()
        self._train_start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.perf_counter() - self._train_start
        self._log_and_print(trainer, "train", elapsed)

    # --- val ---
    def on_validation_epoch_start(self, trainer, pl_module):
        self._reset()
        self._val_start = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module):
        elapsed = time.perf_counter() - self._val_start
        self._log_and_print(trainer, "val", elapsed)
