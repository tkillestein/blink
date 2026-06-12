import gc
from typing import Any

import torch
import wandb
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer
from loguru import logger

from blink.augmentations import (
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
    emit_metric: str | None = "probe/realbogus_logistic_auc",
    group_name: str | None = None,
) -> float | None:

    checkpoint_dir = config.output_dir / "checkpoints"
    log_dir = config.output_dir / "logs"

    logger.debug("Initializing W&B sink")
    wandb_logger = WandbLogger(
        project="blink",
        name=config.experiment.experiment_name,
        save_dir=log_dir,
        group=group_name,
    )

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
        gpu_transform=augmenter,
    )

    model = LeJEPAPretrainer(config)

    _verbose_output = verbose > 0

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
        logger=wandb_logger,
        enable_progress_bar=False,
        enable_model_summary=False,
        profiler="simple",
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        weights_only=False,
    )

    try:
        return (
            float(trainer.callback_metrics[emit_metric].item()) if emit_metric else None
        )

    finally:
        logger.info("Tearing down experiment")
        wandb_logger.experiment.finish()  # Suspect this is a no-op
        wandb.finish()

        trainer.strategy.teardown()
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()
