from typing import Any

import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer

from blink.augmentations import (
    BASE_TRANSFORM,
    MultiCropTransform,
)
from blink.config import LeJEPAPretrainConfig
from blink.data import TrainWebDatasetModule
from blink.model import LeJEPAPretrainer
from blink.probe import ModelProbeCallback
from blink.utils import PHYSICAL_CORES

# For CPU (probably also helps dataloader)
torch.set_num_threads(PHYSICAL_CORES)

# For the GPU
torch.set_float32_matmul_precision("high")


def pretrain(
    config: LeJEPAPretrainConfig,
    verbose: int = 2,
    extra_callbacks: list[Any] | None = None,
    emit_metrics: str | None = "probe/realbogus_logistic_auc",
) -> float | None:

    if extra_callbacks is None:
        extra_callbacks = []

    probe_callback = ModelProbeCallback(
        cfg=config,
        probe_labels=["mag", "realbogus"],
        probe_size=2000,
        every_n_epochs=1,
    )

    augmenter = MultiCropTransform(cfg=config.aug)

    datamodule = TrainWebDatasetModule(
        data_config=config.data,
        cpu_transform=BASE_TRANSFORM,
        gpu_transform=augmenter,
    )

    model = LeJEPAPretrainer(config)

    _verbose_output = verbose > 0

    checkpoint_dir = config.output_dir / "checkpoints"
    log_dir = config.output_dir / "logs"

    trainer = Trainer(
        max_epochs=config.optim.max_epochs,
        precision=config.hardware.precision,
        accelerator=config.hardware.device_type,
        deterministic=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor="probe/mag_linear_r2",
                mode="max",
                save_top_k=3,
                save_last="link",
            ),
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(
                monitor="train/loss_epoch",
                mode="min",
                patience=5,
            ),
            probe_callback,
            *extra_callbacks,
        ],
        logger=[
            WandbLogger(
                project="blink",
                name=config.experiment.experiment_name,
                save_dir=log_dir,
            ),
        ],
        enable_progress_bar=False,
        enable_model_summary=False,
        profiler="simple",
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        weights_only=False,
    )

    if emit_metrics:
        return float(trainer.callback_metrics[emit_metrics].item())

    return None
