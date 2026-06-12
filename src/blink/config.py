from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Self

import rtoml
from loguru import logger
from numpy import ndarray
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PlainSerializer,
    PostgresDsn,
    RootModel,
    SecretStr,
    SkipValidation,
    computed_field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import nn

from blink import __version__ as blink_version
from blink.utils import PHYSICAL_CORES, _default_dataset_name, get_experiment_name
from blink.visual import percentile_asinh_norm, percentile_norm, zscale_interval_norm

SerializablePath = Annotated[
    Path, PlainSerializer(lambda x: str(x.resolve()), return_type=str)
]


# --------------------------------------------------------------------------------------
# Base settings
# --------------------------------------------------------------------------------------


class BlinkSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent.parent / ".env",
        case_sensitive=False,
        env_file_encoding="utf-8",
    )

    pipeline_dburi: PostgresDsn | None = Field(
        default=None, description="DBURI for GOTO pipeline DB"
    )
    wandb_api_key: SecretStr | None = Field(
        default=None,
        description="W&B API key",
    )

    postgres_user: str | None = None
    postgres_password: SecretStr | None = None
    postgres_db: str | None = None
    postgres_port: int | None = 5432
    postgres_host: str | None = "localhost"


# --------------------------------------------------------------------------------------
# Base models
# --------------------------------------------------------------------------------------


class TOMLExportableModel(BaseModel):
    """Base class for TOML exportable models."""

    def to_toml(self) -> str:
        return rtoml.dumps(
            self.model_dump(
                mode="json",
                exclude_none=False,
                exclude_unset=False,
                exclude_defaults=False,
            ),
            pretty=True,
        )

    def write_toml(self, output_path: str | Path) -> None:
        with Path(output_path).open("w") as f:
            f.write(self.to_toml())

    @classmethod
    def _to_blank_toml(cls) -> str:
        return cls.empty().to_toml()

    @classmethod
    def write_blank_toml(cls, output_path: str | Path) -> None:
        with Path(output_path).open("w") as f:
            f.write(cls._to_blank_toml())

    @classmethod
    def from_toml(cls, input_path: Path | str) -> Self:
        with Path(input_path).open("r") as f:
            return cls.model_validate(rtoml.load(f, none_value="null"))

    @classmethod
    def empty(cls) -> Self:
        return cls.model_construct()


# --------------------------------------------------------------------------------------
# User configs
# --------------------------------------------------------------------------------------


class AugmentationConfig(TOMLExportableModel):
    model_type: Literal["augmentation"] = "augmentation"
    global_size: NonNegativeInt = 64
    global_scale_min: float = 0.4
    global_scale_max: float = 0.85
    local_size: NonNegativeInt = 64
    local_scale_min: float = 0.25
    local_scale_max: float = 0.5
    n_local: NonNegativeInt = 8
    n_global: NonNegativeInt = 2
    blur_sigma: NonNegativeFloat = 0.5
    noise_sigma: NonNegativeFloat = 0.02

    @property
    def global_scale(self) -> tuple[float, float]:
        return self.global_scale_min, self.global_scale_max

    @property
    def local_scale(self) -> tuple[float, float]:
        return self.local_scale_min, self.local_scale_max


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


class ViTConfig(TOMLExportableModel):
    """Config class for Vision Transformer models."""

    model_type: Literal["vit"] = "vit"
    model_name: str = "vit_tiny_patch16_224"
    patch_size: NonNegativeInt = 8
    depth: NonNegativeInt = 12
    embed_dim: NonNegativeInt = 243
    num_heads: NonNegativeInt = 3

    def build(self) -> nn.Module:
        """Build any transformer model from `timm`"""
        logger.debug(f"Building model `{self.model_name}` from config")
        import timm

        return timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=0,
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )


class CNNConfig(TOMLExportableModel):
    """Config class for CNN models."""

    model_type: Literal["cnn"] = "cnn"
    model_name: str = "resnet18d"

    embed_dim: NonNegativeInt = 256

    def build(self) -> nn.Module:
        """Build any CNN-family model from `timm`"""
        logger.debug(f"Building model `{self.model_name}` from config")
        import timm

        return timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=self.embed_dim,
            global_pool="avg",
        )


BackboneConfig = Annotated[ViTConfig | CNNConfig, Field(discriminator="model_type")]


# --------------------------------------------------------------------------------------
# JEPA/Objectives
# --------------------------------------------------------------------------------------


class JEPALossConfig(TOMLExportableModel):
    """Config class for JEPA loss objective."""

    n_points: NonNegativeInt = 17
    n_slices: NonNegativeInt = 1024
    lam: NonNegativeFloat = 0.05


class OptimizerConfig(TOMLExportableModel):
    model_type: Literal["optimizer"] = "optimizer"

    learning_rate: NonNegativeFloat = 3e-4
    weight_decay: NonNegativeFloat = 1e-5

    max_epochs: NonNegativeInt = 100


class DataConfig(TOMLExportableModel):
    model_type: Literal["data"] = "data"
    data_dir: SerializablePath = Path()

    # Dataloader params
    load_into_ram: bool = True
    num_workers: NonNegativeInt = min(PHYSICAL_CORES // 2, 8)
    batch_size: NonNegativeInt = 128
    drop_last: bool = True
    persistent_workers: bool = True
    prefetch_factor: NonNegativeInt | None = 2
    shuffle_buffer: NonNegativeInt = 1000
    random_seed: int = 42


class HardwareConfig(TOMLExportableModel):
    model_type: Literal["hardware"] = "hardware"
    device_type: Literal["cpu", "gpu"] = "cpu"
    num_devices: NonNegativeInt = 1
    precision: Literal["32", "16-mixed", "bf16-mixed"] = Field(
        default="32", description="Precision for model training"
    )


class ExperimentConfig(TOMLExportableModel):
    base_dir: SerializablePath = Path()
    experiment_name: str = Field(default_factory=get_experiment_name)

    @property
    def output_dir(self) -> Path:
        return self.base_dir / self.experiment_name


class LeJEPAPretrainConfig(TOMLExportableModel):
    job_type: Literal["pretrain"] = "pretrain"
    backbone: BackboneConfig
    loss: JEPALossConfig
    aug: AugmentationConfig
    optim: OptimizerConfig
    data: DataConfig
    hardware: HardwareConfig
    experiment: ExperimentConfig

    @classmethod
    def empty(cls) -> "LeJEPAPretrainConfig":
        return cls.model_construct(
            backbone=CNNConfig(),
            loss=JEPALossConfig(),
            aug=AugmentationConfig(),
            optim=OptimizerConfig(),
            data=DataConfig(),
            hardware=HardwareConfig(),
            experiment=ExperimentConfig(),
        )

    @property
    def output_dir(self) -> Path:
        return self.experiment.output_dir


# --------------------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------------------


class NullNorm(TOMLExportableModel):
    preprocessor_name: Literal["null_norm"] = "null_norm"

    def __call__(self, input_stamp: ndarray) -> ndarray:
        return input_stamp


class PlainNorm(TOMLExportableModel):
    preprocessor_name: Literal["plain_norm"] = "plain_norm"
    rescale_difference: bool = Field(True, description="Rescale difference to [-1, 1]?")

    def __call__(self, input_stamp: ndarray) -> ndarray:
        return percentile_norm(
            input_stamp,
            min_percent=0,
            max_percent=100,
            rescale_difference=self.rescale_difference,
        )


class PercentileNorm(TOMLExportableModel):
    preprocessor_name: Literal["percentile_norm"] = "percentile_norm"
    min_percent: float = Field(
        0.3, description="Minimum percentile to display", ge=0, le=100
    )
    max_percent: float = Field(
        99.7, description="Maximum percentile to display", ge=0, le=100
    )
    rescale_difference: bool = Field(True, description="Rescale difference to [-1, 1]?")

    def __call__(self, input_stamp: ndarray) -> ndarray:
        return percentile_norm(
            input_stamp,
            min_percent=self.min_percent,
            max_percent=self.max_percent,
            rescale_difference=self.rescale_difference,
        )


class PercentileAsinhNorm(TOMLExportableModel):
    preprocessor_name: Literal["percentile_asinh_norm"] = "percentile_asinh_norm"
    min_percent: float = Field(
        0.3, description="Minimum percentile to display", ge=0, le=100
    )
    max_percent: float = Field(
        99.7, description="Maximum percentile to display", ge=0, le=100
    )
    asinh_stretch: float = Field(
        0.2, description="Strength of asinh stretch to apply", gt=0
    )
    rescale_difference: bool = Field(True, description="Rescale difference to [-1, 1]?")

    def __call__(self, input_stamp: ndarray) -> ndarray:
        return percentile_asinh_norm(
            input_stamp,
            min_percent=self.min_percent,
            max_percent=self.max_percent,
            rescale_difference=self.rescale_difference,
            asinh_stretch=self.asinh_stretch,
        )


class ZScaleIntervalNorm(TOMLExportableModel):
    preprocessor_name: Literal["zscale_norm"] = "zscale_norm"
    contrast: float = Field(0.25, description="ZScaleInterval contrast", gt=0)
    rescale_difference: bool = Field(True, description="Rescale difference to [-1, 1]?")

    def __call__(self, input_stamp: ndarray) -> ndarray:
        return zscale_interval_norm(
            input_stamp, rescale_difference=self.rescale_difference
        )


PreprocessorDiscrim = Annotated[
    NullNorm | PercentileNorm | PlainNorm | PercentileAsinhNorm | ZScaleIntervalNorm,
    Field(discriminator="preprocessor_name"),
]


class ETLConfig(TOMLExportableModel):
    job_type: Literal["etl"] = "etl"
    dataset_name: str = Field(
        description="Extraction job name", default_factory=_default_dataset_name
    )
    stamp_size: int = Field(64, description="Size of stamps to extract")
    layers_to_extract: tuple[str, ...] = Field(
        default=("SCIENCE", "TEMPLATE", "DIFFERENCE"),
    )
    dtype: Literal["float16", "float32", "float64"] = Field(
        "float16",
        description="Dtype to use with zarr storage backend",
    )
    shard_size: int = Field(
        1_000_000_000, description="Target shard size for WebDataset shards"
    )
    num_processes: SkipValidation[int] = Field(
        default=PHYSICAL_CORES // 2,
        le=PHYSICAL_CORES,
        ge=0,
        description="Number of processes to use",
    )
    preprocessor: PreprocessorDiscrim = Field(
        default_factory=ZScaleIntervalNorm,
        description="Stamp scaler to use",
    )

    manifest_path: SerializablePath = Field(
        Path("manifest.csv.zstd"),
        description="Path to manifest to extract",
    )
    output_dir: SerializablePath = Field(
        Path(),
        description="Root directory to build dataset in",
    )

    @property
    def num_layers(self) -> int:
        return len(self.layers_to_extract)

    @classmethod
    def empty(cls) -> "ETLConfig":
        return cls.model_construct(preprocessor=ZScaleIntervalNorm.empty())


class ManifestConfig(TOMLExportableModel):
    job_type: Literal["manifest"] = "manifest"
    date_min: datetime = datetime(2025, 1, 1, tzinfo=UTC)
    date_max: datetime = datetime(2026, 1, 1, tzinfo=UTC)
    number_per_night: int = 1000
    max_rows: NonNegativeInt = 500000
    db_threads: NonNegativeInt = 4

    output_name: str = Field(default_factory=get_experiment_name)
    output_root: SerializablePath = Field(default_factory=Path.cwd)

    @computed_field
    @property
    def output_dir(self) -> Path:
        return self.output_root.resolve() / self.output_name


BlinkConfig = Annotated[
    LeJEPAPretrainConfig | ETLConfig | ManifestConfig,
    Field(discriminator="job_type"),
]


class BlinkConfigFile(RootModel):
    root: BlinkConfig

    @classmethod
    def from_toml(cls, path: Path) -> "BlinkConfigFile":
        return cls.model_validate(rtoml.load(path, none_value="null"))


def grab_timestamp() -> datetime:
    return datetime.now(UTC)


class ETLResultConfig(TOMLExportableModel):
    etl_config: ETLConfig
    num_stamps: NonNegativeInt
    created_at: datetime = Field(default_factory=grab_timestamp)
    blink_version: str = Field(default=blink_version)
    shard_count: NonNegativeInt

    # Optionals for rescaling
    channel_means: list[float] = [0.0, 0.0, 0.0]
    channel_stds: list[float] = [1.0, 1.0, 1.0]
