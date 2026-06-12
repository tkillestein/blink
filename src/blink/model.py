import lejepa.multivariate
import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.types import (
    OptimizerLRScheduler,
)
from loguru import logger

from blink.config import LeJEPAPretrainConfig


class LeJEPAPretrainer(LightningModule):
    def __init__(
        self,
        config: LeJEPAPretrainConfig | None,
        config_json: str | None = None,
    ) -> None:
        super().__init__()

        logger.debug("Initialising LeJEPAPretrainer from config")
        if config is None:
            logger.debug("Loading config from JSON if exists")
            config = (
                LeJEPAPretrainConfig.model_validate_json(config_json)
                if config_json is not None
                else LeJEPAPretrainConfig.empty()
            )

        self.cfg = config

        logger.debug("Saving hyperparameters to logger")
        self.save_hyperparameters(
            {
                "config_json": self.cfg.model_dump_json(),
                **self.cfg.backbone.model_dump(mode="json", exclude={"model_type"}),
                **self.cfg.loss.model_dump(mode="json", exclude={"model_type"}),
                **self.cfg.optim.model_dump(mode="json", exclude={"model_type"}),
                **self.cfg.aug.model_dump(mode="json", exclude={"model_type"}),
                "batch_size": self.cfg.data.batch_size,
                "name": self.cfg.experiment.experiment_name,
            }
        )

        # Build model
        self.backbone = self.cfg.backbone.build()

        # Build sigreg loss using form in paper
        self.sigreg_loss = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=lejepa.univariate.EppsPulley(
                n_points=self.cfg.loss.n_points
            ),
            num_slices=self.cfg.loss.n_slices,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        global_views = batch[: self.cfg.aug.n_global]
        local_views = batch[self.cfg.aug.n_global :]

        # All views through the same encoder, ALL with gradients (no stop-grad)
        proj_locals = self(torch.cat(local_views, dim=0))
        proj_globals = self(torch.cat(global_views, dim=0))

        local_embeddings = proj_locals.chunk(self.cfg.aug.n_local, dim=0)
        global_embeddings = proj_globals.chunk(self.cfg.aug.n_global, dim=0)

        # Compute embedding stats every 50 steps
        if self.global_step % 50 == 0:
            self._log_embedding_stats(proj_locals.detach(), "local")
            self._log_embedding_stats(proj_globals.detach(), "global")

        # Predict the global-view CENTRE from each local view
        global_center = torch.stack(global_embeddings, dim=0).mean(dim=0)
        mse_loss = torch.stack(
            [F.mse_loss(loc, global_center) for loc in local_embeddings]
        ).mean()

        all_embeddings = torch.cat([proj_locals, proj_globals], dim=0)
        sreg_loss = self.sigreg_loss(all_embeddings)

        loss = mse_loss * (1 - self.cfg.loss.lam) + sreg_loss * self.cfg.loss.lam

        self.log_dict(
            {
                "train/mse_loss": mse_loss,
                "train/sreg_loss": sreg_loss,
                "train/loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log_dict({"hp_metric": loss}, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        logger.debug("Configuring optimisers")
        scaled_lr = self.cfg.optim.learning_rate * (self.cfg.data.batch_size / 256)
        logger.debug(
            f"Base LR: {self.cfg.optim.learning_rate:.2e} -> Scaled LR: {scaled_lr:.2e}"
        )

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=scaled_lr,
            weight_decay=self.cfg.optim.weight_decay,
        )

        total_epochs = self.cfg.optim.max_epochs
        warmup_epochs = max(5, int(total_epochs * 0.05))

        logger.debug(
            f"Learning rate schedule: {warmup_epochs} epoch warmup."
            f" Total {total_epochs} epochs."
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scaled_lr,
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=warmup_epochs / total_epochs,  # warm-up fraction matches original
            anneal_strategy="cos",  # cosine decay, matches original
            div_factor=10.0,  # start at max_lr / 10  (≈ start_factor=0.1)
            final_div_factor=100.0,  # end at max_lr / 1000... see note below
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # <-- per-batch stepping
                "frequency": 1,
            },
        }

    @torch.no_grad()
    def _log_embedding_stats(self, emb: torch.Tensor, tag: str) -> None:
        emb = emb.float()
        std = emb.std(dim=0).mean()

        c = emb - emb.mean(dim=0, keepdim=True)
        cov = (c.T @ c) / (emb.shape[0] - 1)
        eff_rank = cov.trace().pow(2) / cov.pow(2).sum().clamp_min(1e-12)  # no eigvalsh

        sub = emb[:512]  # cap O(m^2) cost
        n = F.normalize(sub, dim=1)
        sim = n @ n.T
        m = sim.shape[0]
        off_diag = (sim.sum() - sim.diagonal().sum()) / (m * (m - 1))

        self.log_dict(
            {
                f"emb/{tag}_std": std,
                f"emb/{tag}_eff_rank": eff_rank,
                f"emb/{tag}_cos_sim": off_diag,
            },
            on_step=True,
            on_epoch=False,
            sync_dist=False,  # diagnostic, skip the allreduce
        )

        del cov, c

    def save_model(self) -> None:
        logger.info("Exporting model to disk")

        onnx_path = self.cfg.experiment.output_dir / "model.onnx"
        self.to_onnx(
            file_path=onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=True,
            opset_version=17,
        )
