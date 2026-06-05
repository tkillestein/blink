from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import psutil
from faker import Faker
from loguru import logger
from sklearn.preprocessing import QuantileTransformer

faker = Faker()

PHYSICAL_CORES = psutil.cpu_count(logical=False)

module_rng = np.random.default_rng(42)


def _default_dataset_name() -> str:
    return f"dataset_{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}"


def get_experiment_name() -> str:
    return "-".join(faker.words(3))


def nightly_dates(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    n_days = (end - start).days
    return [
        (start + timedelta(days=i), start + timedelta(days=i + 1))
        for i in range(n_days)
    ]


def diversity_sample_from_df(
    source_table: pl.DataFrame,
    k: int,
    features_to_sample: list[str],
    quantile_threshold: int = 1000,
) -> pl.DataFrame:
    if source_table.height <= quantile_threshold:
        logger.warning(f"Quantile transforming with only {source_table.height} items")

    feature_array = source_table.select(features_to_sample).to_numpy()
    features_prescaled = (
        QuantileTransformer(
            output_distribution="uniform",
            n_quantiles=min(1000, feature_array.shape[0]),
        )
        .fit_transform(feature_array)
        .astype("float32")
    )

    # NB: since distance computation is the hot path, rather than calling norm directly,
    # Use vector product identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2xy to reduce work
    # to a single fused multiply-add for maximum speed
    sq_norms = np.einsum("ij,ij->i", features_prescaled, features_prescaled)

    # Heuristic 2-approximation: get a random point, obtain the furthest point from that
    # and then the furthest point from that as our init. This stabilises the iterations.
    r_idx = module_rng.integers(features_prescaled.shape[0])
    dists_from_r = (
        sq_norms + sq_norms[r_idx] - 2 * features_prescaled @ features_prescaled[r_idx]
    )
    a_idx = np.argmax(dists_from_r)

    dists_from_a = (
        sq_norms + sq_norms[a_idx] - 2 * features_prescaled @ features_prescaled[a_idx]
    )
    b_idx = np.argmax(dists_from_a)

    dists_from_b = (
        sq_norms + sq_norms[b_idx] - 2 * features_prescaled @ features_prescaled[b_idx]
    )

    selected = [a_idx, b_idx]
    min_dist = np.minimum(dists_from_a, dists_from_b)
    min_dist[a_idx] = -1
    min_dist[b_idx] = -1

    # Now get best option at each step greedily
    # Note: keeping min_dist transforms this from O(N^2) to O(N).
    for _ in range(2, k):
        # Select the point with maximum min-distance
        new_idx = np.argmax(min_dist)
        selected.append(new_idx)

        dists_sq = (
            sq_norms
            + sq_norms[new_idx]
            - 2.0 * features_prescaled @ features_prescaled[new_idx]
        )
        min_dist = np.minimum(min_dist, dists_sq)
        min_dist[new_idx] = -1

    return source_table[np.array(selected)]
