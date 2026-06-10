from collections.abc import Iterable, Sequence
from itertools import islice
from typing import cast

import numpy as np
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from loguru import logger
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from webdataset.compat import WebDataset

from blink.config import LeJEPAPretrainConfig


class ModelProbeCallback(Callback):
    def __init__(
        self,
        cfg: LeJEPAPretrainConfig,
        probe_labels: Sequence[str],
        probe_size: int = 5000,
        every_n_epochs: int = 3,
        classification_targets: tuple[str, ...] = ("realbogus",),
        cv_folds: int = 5,
    ) -> None:
        self.probe_labels = probe_labels
        self.classification_targets = classification_targets
        self._probe_meta = None
        self._probe_images = None
        self.every_n_epochs = every_n_epochs
        self.cfg = cfg
        self.probe_size = probe_size
        self.cv_folds = cv_folds

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:

        # We don't want to run the probe on all machines
        if trainer.global_rank != 0:
            return

        chunks = sorted((self.cfg.data.data_dir / "webdataset_store").glob("*.tar"))
        chunk_urls = [str(chunk.resolve()) for chunk in chunks]

        probe_dataset = (
            WebDataset(
                chunk_urls,
                shardshuffle=False,  # Keep it deterministic for same dataset
            )
            .decode()
            .to_tuple("stamp.npy", "meta.json")
            .map(lambda t: ((torch.from_numpy(t[0])).to(torch.float32), t[1]))
        )

        # Keep the type checker happy
        probe_dataset = cast("Iterable", probe_dataset)

        logger.debug(f"Building probe dataset of {self.probe_size} examples")
        all_images, all_meta = [], []
        for image, meta in islice(probe_dataset, self.probe_size):
            all_images.append(image)
            all_meta.append(meta)

        self._probe_images = torch.stack(all_images)
        self._probe_meta = all_meta

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._probe_images is None or self._probe_meta is None:
            msg = (
                "probe_dataloader not initialised — did `on_fit_start` fire correctly?"
            )
            raise RuntimeError(msg)

        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        pl_module.eval()
        with torch.no_grad():
            batch_size = self.cfg.data.batch_size
            embeddings = torch.cat(
                [
                    pl_module(
                        self._probe_images[i : i + batch_size].to(pl_module.device)
                    ).cpu()
                    for i in range(0, len(self._probe_images), batch_size)
                ]
            ).numpy()
        pl_module.train()

        targets = {
            k: np.array([m[k] for m in self._probe_meta]) for k in self.probe_labels
        }

        alpha_range = np.logspace(-4, 4, 10)

        for target_name, target_val in targets.items():
            if target_name in self.classification_targets:
                threshold_point = 0.5

                binarised_labels = target_val > threshold_point

                base_lr_classifier = LogisticRegressionCV(
                    Cs=alpha_range,
                    scoring="roc_auc",
                    l1_ratios=(0.0,),
                    max_iter=10000,
                    use_legacy_attributes=False,
                    class_weight="balanced",
                    n_jobs=-1,
                    verbose=0,
                )

                scores = cross_val_score(
                    estimator=base_lr_classifier,
                    X=embeddings,
                    y=binarised_labels,
                    cv=self.cv_folds,
                )

                pl_module.log(f"probe/{target_name}_logistic_auc", scores.mean())

                # K-neighbours
                base_clf = KNeighborsClassifier(n_neighbors=3)
                cv_score = cross_val_score(
                    base_clf,
                    X=embeddings,
                    y=binarised_labels,
                    n_jobs=-1,
                    scoring="roc_auc",
                    verbose=0,
                    cv=self.cv_folds,
                )
                pl_module.log(f"probe/{target_name}_knn_auc", cv_score.mean())
            else:
                # Linear probe
                linear_probe = RidgeCV(
                    alphas=alpha_range,
                    scoring="r2",
                    cv=None,  # Use GCV to tune alpha internally
                )

                scores = cross_val_score(
                    estimator=linear_probe,
                    X=embeddings,
                    y=target_val,
                    cv=self.cv_folds,
                    scoring="r2",
                )

                pl_module.log(
                    f"probe/{target_name}_linear_r2",
                    scores.mean(),
                )

                # KNN probe
                base_clf = KNeighborsRegressor(n_neighbors=3)
                cv_score = cross_val_score(
                    base_clf,
                    X=embeddings,
                    y=target_val,
                    n_jobs=-1,
                    scoring="r2",
                    cv=self.cv_folds,
                )
                pl_module.log(f"probe/{target_name}_knn_r2", cv_score.mean())
