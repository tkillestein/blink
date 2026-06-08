from collections.abc import Sequence
from pathlib import Path

import numpy as np
from astropy.visualization import ZScaleInterval
from matplotlib import pyplot as plt


def percentile_norm(
    input_stamp: np.ndarray,
    min_percent: float = 0.5,
    max_percent: float = 99.5,
    rescale_difference: bool = True,
) -> np.ndarray:
    """
    Normalise a batch of stamps by their upper and lower percentiles.

    This function is safe and automatically accounts for the floating-point precision of
    the input stamps. The result is also computed channel-wise.

    Args:
        input_stamp: Batch of stamps, with layout NCHW
        min_percent: Minimum percentile to display
        max_percent: Maximum percentile to display
        rescale_difference: If true, difference channel is rescaled to range [-1, 1]

    Returns:
        Batch of stamps normalised between 0 and 1

    """
    epsilon = float(np.finfo(input_stamp.dtype).eps)

    lo, hi = np.nanpercentile(
        input_stamp, [min_percent, max_percent], axis=(2, 3), keepdims=True
    )

    clip_stamp = np.clip(input_stamp, a_min=lo, a_max=hi)
    scaled_stamp = (clip_stamp - lo) / (hi - lo + epsilon)

    if rescale_difference:
        scaled_stamp[:, 2, :, :] = scaled_stamp[:, 2, :, :] * 2 - 1

    return scaled_stamp


def plain_norm(input_stamp: np.ndarray, rescale_difference: bool = True) -> np.ndarray:
    """
    Normalise stamps by mapping min/max values to [0, 1]

    Args:
        input_stamp: Batch of stamps, with layout NCHW
        rescale_difference: If true, difference channel is rescaled to range [-1, 1]

    Returns:
        Batch of stamps normalised between 0 and 1

    """
    return percentile_norm(
        input_stamp,
        min_percent=0,
        max_percent=100,
        rescale_difference=rescale_difference,
    )


def percentile_asinh_norm(
    input_stamp: np.ndarray,
    min_percent: float = 0.5,
    max_percent: float = 99.5,
    asinh_stretch: float = 0.2,
    rescale_difference: bool = True,
) -> np.ndarray:
    """
    Normalise a batch of stamps by their upper and lower percentiles, and apply an asinh
    stretch.

    Args:
        input_stamp: Batch of stamps, with layout NCHW
        min_percent: Minimum percentile to display
        max_percent: Maximum percentile to display
        rescale_difference: If true, difference channel is rescaled to range [-1, 1]
        asinh_stretch: Power of asinh stretch

    Returns:
        Batch of stamps normalised between 0 and 1, with asinh stretch applied

    """
    scaled_stamps = percentile_norm(
        input_stamp, min_percent, max_percent, rescale_difference
    )

    return np.asinh(scaled_stamps / asinh_stretch) / np.asinh(1 / asinh_stretch)


def zscale_interval_norm(
    input_stamp: np.ndarray,
    contrast: float = 0.25,
    rescale_difference: bool = True,
) -> np.ndarray:

    vmin, vmax = ZScaleInterval(contrast=contrast).get_limits(
        input_stamp.astype("float")  # Cast needed in case we have float16 type
    )
    scaled_stamp = input_stamp - vmin / (vmax - vmin)

    if rescale_difference:
        scaled_stamp[:, 2, :, :] = scaled_stamp[:, 2, :, :] * 2 - 1

    return scaled_stamp


def plot_triplets(
    stamps: np.ndarray,
    output_path: Path,
    axis_names: Sequence[str] = ("SCIENCE", "TEMPLATE", "DIFFERENCE"),
    title: str | None = None,
) -> None:

    fig, axes = plt.subplots(stamps.shape[0], 3, figsize=(6, 6), dpi=240)
    plt.subplots_adjust(hspace=0.05, wspace=-0.05)

    for buf_idx, stamp in enumerate(stamps):
        for ax_idx, channel in enumerate(stamp):
            axes[buf_idx, ax_idx].imshow(channel, cmap="bone")

    # Post-config
    for ax_idx, name in enumerate(axis_names):
        axes[0][ax_idx].set_title(name)

    for ax in axes.ravel():
        ax.axis("off")

    if title:
        fig.suptitle(title, fontweight="bold")

    fig.savefig(fname=output_path, format="png", bbox_inches="tight", pad_inches=0.05)
