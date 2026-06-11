from functools import partial
from pathlib import Path

import optuna
from optuna import samplers
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from blink.config import (
    AugmentationConfig,
    CNNConfig,
    DataConfig,
    ExperimentConfig,
    HardwareConfig,
    JEPALossConfig,
    LeJEPAPretrainConfig,
    OptimizerConfig,
)
from blink.pretrain import pretrain

MAX_EPOCHS = 25
ROOT_EXPERIMENT_DIR = Path(
    "/springbrook/share/physics/phsrcc/blink_data/resnet_sweep_25epochs"
)
GROUP_NAME = "2026-06-11_resnet_sweep"


def lejepa_hyperparam_objective(trial: optuna.Trial, metric_to_trace: str) -> float:

    weight_decay = trial.suggest_float("weight_decay", 1e-5, 0.05, log=True)
    base_lr = trial.suggest_float("learning_rate", low=1e-5, high=1e-3, log=True)
    batch_size = 2 ** trial.suggest_int("batch_size", 6, 10)
    embedding_dim = 2 ** trial.suggest_int("embedding_dim", 6, 10)

    optim = OptimizerConfig(
        learning_rate=base_lr,
        weight_decay=weight_decay,
        max_epochs=MAX_EPOCHS,
    )

    backbone = CNNConfig(
        model_name="resnet18d",
        embed_dim=embedding_dim,
    )

    data = DataConfig(
        batch_size=batch_size,
        data_dir=Path(ROOT_EXPERIMENT_DIR),
    )

    loss = JEPALossConfig()

    populated_config = LeJEPAPretrainConfig(
        optim=optim,
        backbone=backbone,
        loss=loss,
        data=data,
        experiment=ExperimentConfig(),
        aug=AugmentationConfig(),
        hardware=HardwareConfig(device_type="gpu", precision="bf16-mixed"),
    )

    result = pretrain(
        config=populated_config,
        emit_metrics=metric_to_trace,
    )

    if result is None:
        msg = "Metrics not properly configured - try again."
        raise RuntimeError(msg)

    return result


if __name__ == "__main__":
    storage = JournalStorage(
        JournalFileBackend(file_path=str(ROOT_EXPERIMENT_DIR / "study.log"))
    )

    study = optuna.create_study(
        study_name="test",
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=samplers.TPESampler(
            multivariate=True,
            group=True,
            constant_liar=True,
            n_startup_trials=10,
        ),
    )

    study.optimize(
        partial(
            lejepa_hyperparam_objective, metric_to_trace="probe/realbogus_logistic_auc"
        ),
        n_trials=100,
    )
