import io
import random
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import torch
import webdataset as wds
from astropy.io import fits
from astropy.nddata import Cutout2D, NoOverlapError
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from loguru import logger
from polars.dataframe.group_by import GroupBy
from threadpoolctl import ThreadpoolController
from torch.utils.data import DataLoader, Dataset
from webdataset.compat import WebDataset
from webdataset.shardlists import split_by_node, split_by_worker

from blink.augmentations import BaseTransform
from blink.config import (
    BlinkSettings,
    DataConfig,
    ETLConfig,
    ETLResultConfig,
    ManifestConfig,
)
from blink.utils import PHYSICAL_CORES, diversity_sample_from_df, nightly_dates

# Pull out constants
WEBDATASET_STORE = "webdataset_store"
STAMP_INDEX_NAME = "stamp_id"

controller = ThreadpoolController()


class TrainWebDatasetModule(LightningDataModule):
    def __init__(
        self,
        data_config: DataConfig,
        gpu_transform: Callable,
    ) -> None:
        super().__init__()

        self.data_config = data_config
        self.gpu_transform = gpu_transform

        dataset_metadata = ETLResultConfig.from_toml(
            self.data_config.data_dir / WEBDATASET_STORE / "dataset_info.toml"
        )

        self.cpu_transform = BaseTransform(
            channel_means=dataset_metadata.channel_means,
            channel_stds=dataset_metadata.channel_stds,
        )

        self.total_stamps = dataset_metadata.num_stamps
        self.batch_size = data_config.batch_size

        self.array_shape = (
            self.batch_size,
            dataset_metadata.etl_config.num_layers,
            dataset_metadata.etl_config.stamp_size,
            dataset_metadata.etl_config.stamp_size,
        )

    def _make_dataset(self) -> Dataset:

        chunks = sorted((self.data_config.data_dir / WEBDATASET_STORE).glob("*.tar"))
        chunk_urls = [str(chunk.resolve()) for chunk in chunks]

        return (
            WebDataset(
                chunk_urls,
                shardshuffle=len(chunk_urls),
                nodesplitter=split_by_node,
                workersplitter=split_by_worker,
                detshuffle=True,
                seed=self.data_config.random_seed,
            )
            .shuffle(size=self.data_config.shuffle_buffer)
            .decode()
            .to_tuple("stamp.npy")
            .map(lambda t: self.cpu_transform(torch.from_numpy(t[0])))
            .batched(
                batchsize=self.data_config.batch_size,
                partial=False,
                collation_fn=torch.stack,
            )
            .with_epoch(nbatches=self.steps_per_epoch)  # ty:ignore[unresolved-attribute]
            .with_length(n=self.steps_per_epoch, silent=True)
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self._make_dataset(),
            batch_size=None,
            num_workers=self.data_config.num_workers,
            prefetch_factor=self.data_config.prefetch_factor,
            persistent_workers=self.data_config.persistent_workers,
        )

    @property
    def steps_per_epoch(self) -> int:
        world_size = self.trainer.world_size if self.trainer else 1
        return self.total_stamps // (self.data_config.batch_size * world_size)

    def on_after_batch_transfer(
        self, batch: torch.Tensor, dataloader_idx: int
    ) -> list[torch.Tensor]:

        return self.gpu_transform(batch)


class WebDatasetIngestPipeline:
    def __init__(
        self,
        etl_config: ETLConfig,
    ) -> None:
        """
        This class has one job - take a CSV file of filepaths and coordinates, and grab
        stamps and metadata. Quickly!
        """
        self.cfg = etl_config

        logger.info(f"Creating worker pool with {self.cfg.num_processes} processes")
        self.executor_pool = ProcessPoolExecutor(
            max_workers=self.cfg.num_processes,
        )

        self.manifest = self.read_manifest()

        projected_stamp_bytes = (
            self.manifest.height
            * np.ones(
                (1, self.cfg.num_layers, self.cfg.stamp_size, self.cfg.stamp_size),
                dtype=self.cfg.dtype,
            ).nbytes
        )

        self.num_shards = max(
            16,  # Set minimum floor for efficient multi-process reads
            int(projected_stamp_bytes // self.cfg.shard_size) + 1,
        )
        logger.info(f"Scattering stamps across {self.num_shards} sharded files")

        self.write_path = self.cfg.output_dir / WEBDATASET_STORE

        if not self.write_path.exists():
            logger.debug("Creating folder structure")
            self.write_path.mkdir()
        else:
            msg = "Dataset already exists in this location"
            raise RuntimeError(msg)

        # Scatter writes among `self.num_shards` files to enforce randomness
        self.writers = [
            wds.writer.TarWriter(str(self.write_path / f"shard_{shard_idx:06d}.tar"))
            for shard_idx in range(self.num_shards)
        ]

        # Initialise online statistics variables
        self.stamp_count = 0

        # Setup for Welford online mean-variance estimation
        self._pixel_count = np.zeros((self.cfg.num_layers,), dtype="int64")
        self._means = np.zeros((self.cfg.num_layers,), dtype="float64")
        self._M2 = np.zeros((self.cfg.num_layers,), dtype="float64")

    def update_batch_statistics(self, stamp_batch: np.ndarray) -> None:
        _n, _c, _h, _w = stamp_batch.shape
        batch_pixel_count = _n * _h * _w

        old_means = self._means.copy()

        # Welford streaming mean-variance estimator, adjusted to count over images
        self._pixel_count += batch_pixel_count
        batch_means = np.mean(stamp_batch, axis=(0, 2, 3))
        self._means += (batch_pixel_count / self._pixel_count) * (
            batch_means - old_means
        )

        old_means_v = old_means.reshape(1, _c, 1, 1)
        new_means_v = self._means.reshape(1, _c, 1, 1)

        self._M2 += np.sum(
            (stamp_batch - old_means_v) * (stamp_batch - new_means_v),
            axis=(0, 2, 3),
        )

    def ingest(self) -> None:
        logger.info(f"Beginning ingest as WebDataset: saving data to {self.write_path}")
        start = perf_counter()

        # Parallelise over common frames
        frame_groups, _num_frames = self.group_manifest("filepath")

        tasks = []
        with self.executor_pool as pool:
            for (filepath,), group_data in frame_groups:
                future = pool.submit(
                    ingest_single_frame,
                    filepath,
                    group_data,
                    self.cfg,
                )
                tasks.append(future)

            try:
                for stamps, metadata in (f.result() for f in as_completed(tasks)):
                    processed = self.cfg.preprocessor(stamps)

                    # Cast directly to desired output dtype
                    processed = processed.astype(self.cfg.dtype)

                    self.update_batch_statistics(processed)

                    for stamp, meta in zip(processed, metadata, strict=True):
                        stamp_id = meta[STAMP_INDEX_NAME]
                        buf = io.BytesIO()
                        np.save(buf, stamp)

                        # Scatter writes across all shards for better randomness
                        writer: wds.writer.TarWriter = random.choice(self.writers)  # noqa: S311
                        writer.write(
                            {
                                "__key__": f"{self.cfg.dataset_name}"
                                f"/stamp_{stamp_id:08d}",
                                "stamp.npy": buf.getvalue(),
                                "meta.json": meta,
                            }
                        )
                        self.stamp_count += 1

                    if self.stamp_count % 100 == 0:
                        perc_complete = self.stamp_count / self.manifest.height * 100
                        logger.info(
                            f"{self.stamp_count} of "
                            f"{self.manifest.height} ({perc_complete:.1f}%) "
                            f"stamps completed"
                        )

            except KeyboardInterrupt:
                logger.warning("Cancelling pending extract tasks")

                self.executor_pool.shutdown(wait=False, cancel_futures=True)
                self.flush_writers()
                raise

            except Exception as e:
                logger.exception(e)
                raise

            finally:
                end = perf_counter()
                self.executor_pool.shutdown(wait=True)
                self.flush_writers()

        logger.info("Ingest completed!")

        runtime = end - start

        logger.info(f"Total stamps in dataset: {self.stamp_count}")
        logger.info(f"Total runtime: {runtime:.1f} seconds")
        logger.info(
            f"Effective throughput: {self.stamp_count / runtime:.1f} stamps/sec"
        )
        logger.info(
            f"Effective throughput (per process): "
            f"{self.stamp_count / runtime / self.cfg.num_processes:.1f} "
            f"stamps/sec/process",
        )

        dataspec_path = self.write_path / "dataset_info.toml"

        spec_out = ETLResultConfig(
            etl_config=self.cfg,
            num_stamps=self.stamp_count,
            shard_count=self.num_shards,
            channel_means=list(self.dataset_mean),
            channel_stds=list(self.dataset_stddev),
        )

        spec_out.write_toml(dataspec_path)

    def flush_writers(self) -> None:
        for writer in self.writers:
            writer.close()

    def read_manifest(self) -> pl.DataFrame:
        logger.info(f"Reading manifest from {self.cfg.manifest_path}")

        manifest = pl.read_csv(self.cfg.manifest_path, try_parse_dates=True)

        logger.info(f"Found {manifest.shape[0]} sources to ingest")
        return manifest

    def group_manifest(self, group_keys: str | list[str]) -> tuple[GroupBy, int]:
        groups = self.manifest.group_by(by=group_keys)
        n_groups = groups.agg(pl.len()).shape[0]
        logger.info(f"Found {n_groups} unique images to parallelise over")
        return groups, n_groups

    @classmethod
    def from_toml(cls, config_path: Path) -> "WebDatasetIngestPipeline":
        _config = ETLConfig.from_toml(config_path)

        return cls(etl_config=_config)

    @property
    def dataset_mean(self) -> np.ndarray:
        return self._means

    @property
    def dataset_variance(self) -> np.ndarray:
        return self._M2 / (self._pixel_count - 1)

    @property
    def dataset_stddev(self) -> np.ndarray:
        return np.sqrt(self._M2 / (self._pixel_count - 1))


def ingest_single_frame(
    input_filepath: str | Path,
    frame_data: pl.DataFrame,
    etl_config: ETLConfig,
) -> tuple[npt.NDArray[Any], list[dict[str, Any]]]:
    cutout_locations = [
        (row["x"], row["y"]) for row in frame_data.iter_rows(named=True)
    ]
    stamps = extract_stamps_from_frame(
        fits_filepath=str(input_filepath),
        locations=cutout_locations,
        stamp_size=etl_config.stamp_size,
        extensions_to_extract=etl_config.layers_to_extract,
        output_dtype=etl_config.dtype,
    )

    # Remove bad stamps entirely: if they aren't modified from the default value.
    bad_indices = np.all(stamps == 0, axis=(1, 2, 3))

    num_valid_stamps = np.sum(~bad_indices)
    num_total_stamps = frame_data.shape[0]

    if num_valid_stamps != num_total_stamps:
        logger.warning(
            f"{num_total_stamps - num_valid_stamps} stamps failed for {input_filepath}"
        )

    good_stamps = stamps[~bad_indices]

    good_metadata = [
        row
        for row, bad in zip(frame_data.iter_rows(named=True), bad_indices, strict=True)
        if not bad
    ]

    logger.debug(
        f"{num_valid_stamps} / {num_total_stamps} "
        f"stamps extracted from {input_filepath}"
    )

    return good_stamps, good_metadata


def extract_stamps_from_frame(
    fits_filepath: str,
    locations: list[tuple[float, float]],
    stamp_size: int,
    extensions_to_extract: tuple[str, ...] = ("SCIENCE", "TEMPLATE", "DIFFERENCE"),
    output_dtype: str = "float16",
) -> npt.NDArray:

    # NCHW for Torch!
    arr_prealloc = np.zeros(
        (
            len(locations),
            len(extensions_to_extract),
            stamp_size,
            stamp_size,
        ),
        dtype=output_dtype,
    )
    with fits.open(
        fits_filepath,
        mode="readonly",
        memmap=True,
        lazy_load_hdus=True,
    ) as hdulist:
        for ext_idx, ext_name in enumerate(extensions_to_extract):
            for position_idx, position in enumerate(locations):
                try:
                    arr_prealloc[position_idx, ext_idx, :, :] = Cutout2D(
                        data=hdulist[ext_name].data,
                        size=(stamp_size, stamp_size),
                        position=position,
                        fill_value=0,
                        mode="partial",
                    ).data.astype(
                        output_dtype,
                    )  # Force cast here, rather than leave it to auto-casting

                except KeyboardInterrupt:
                    raise

                except NoOverlapError:
                    logger.debug(f"Position {position} has no overlap with {ext_name}")
                    continue

                except (KeyError, IndexError):
                    logger.warning(
                        f"Extension {ext_name} missing from FITS file {fits_filepath}"
                        f" - skipping entire file"
                    )
                    break

                except Exception as e:
                    msg = (
                        "Stamp extraction failed for {fits_filepath} with exception {e}"
                    )
                    raise RuntimeError(msg) from e

    return arr_prealloc


def gather_manifest(manifest_config: ManifestConfig) -> pl.DataFrame:
    """Create a data ingest manifest"""
    if BlinkSettings().pipeline_dburi is None:
        msg = "Pipeline database URI is required - ensure `.env` is populated"
        raise RuntimeError(msg)

    logger.info("Generating manifest")
    logger.info(manifest_config.model_dump_json(indent=2))

    pipeline_dburi = str(BlinkSettings().pipeline_dburi)
    nightly_ranges = nightly_dates(manifest_config.date_min, manifest_config.date_max)

    query_to_param = """
                     SELECT psd.ra,
                            psd.dec,
                            psd.x,
                            psd.y,
                            psd.mag,
                            psd.mag_uncert,
                            psd.photom_flag,
                            psd.realbogus,
                            psd.image_id,
                            s.filepath,
                            psd.hfd,
                            s.tel,
                            s.ut
                     FROM photometry.set_difference psd
                              JOIN image.set s ON s.id = psd.image_id
                     WHERE psd.date_mid >= $1 \
                       AND psd.date_mid < $2
                         LIMIT $3 \
                     """

    def fetch_rows(start_time: datetime, end_time: datetime) -> pl.DataFrame | None:
        nightly_df = pl.read_database_uri(
            query=query_to_param,
            uri=pipeline_dburi,
            engine="adbc",
            execute_options={
                "parameters": (start_time, end_time, manifest_config.max_rows)
            },
            schema_overrides={
                "tel": pl.UInt8,
                "ut": pl.UInt8,
                "photom_flag": pl.UInt16,
                "filepath": pl.String,
                "ra": pl.Float32,
                "dec": pl.Float32,
                "image_id": pl.UInt32,
            },
        )
        logger.debug(
            f"Fetched {nightly_df.shape[0]} rows between {start_time} and {end_time}"
        )

        if nightly_df.height < 4 * manifest_config.number_per_night:
            logger.warning("Not enough detections to diversity-sample - skipping")
            return None

        # Cut off thread contention at source
        # NB: because of GIL + hyperthreading this is actually a bit conservative.
        with controller.limit(limits=PHYSICAL_CORES // manifest_config.db_threads):
            return diversity_sample_from_df(
                nightly_df,
                k=manifest_config.number_per_night,
                features_to_sample=[
                    "ra",
                    "dec",
                    "x",
                    "y",
                    "mag",
                    "mag_uncert",
                    "realbogus",
                    "hfd",
                    "tel",
                    "ut",
                ],
            )

    logger.debug(f"Spawning thread pool with {manifest_config.db_threads} workers")
    try:
        with ThreadPoolExecutor(max_workers=manifest_config.db_threads) as executor:
            futures = {
                executor.submit(fetch_rows, s, e): (s, e) for s, e in nightly_ranges
            }
            samples = [
                f.result() for f in as_completed(futures) if f.result() is not None
            ]
    except KeyboardInterrupt:
        logger.warning("Received shutdown signal - cancelling tasks")
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    diverse_dataframe = pl.concat(samples)

    output_path = manifest_config.output_dir
    _path = (
        output_path / f"manifest_{manifest_config.date_min.date()}"
        f"-{manifest_config.date_max.date()}"
        f"_{diverse_dataframe.height}_examples.csv.zst"
    )

    logger.debug(f"Writing manifest to {_path}")
    diverse_dataframe.with_row_index(name=STAMP_INDEX_NAME).write_csv(
        _path, compression="zstd"
    )

    return diverse_dataframe


def collate_img_and_meta(batch: list) -> tuple[torch.Tensor, list[dict]]:
    images = torch.stack([item[0] for item in batch])
    metadata = [item[1] for item in batch]  # list of dicts
    return images, metadata
