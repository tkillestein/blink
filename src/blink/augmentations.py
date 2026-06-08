from typing import Any

import torch
import torchvision.transforms.v2 as T

from blink.config import AugmentationConfig


class SmartPSFBlur(T.Transform):
    def __init__(
        self,
        sigma_range: tuple[float, float] = (0.1, 0.5),
        ellipticity_range: tuple[float, float] = (0.1, 0.5),
    ) -> None:
        self.ellipticity_range = ellipticity_range
        self.sigma_range = sigma_range

        super().__init__()

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:

        sigma = (torch.rand((1,)) + self.sigma_range[0]) * (
            self.sigma_range[1] - self.sigma_range[0]
        )
        ellipticity = (torch.rand((1,)) + self.ellipticity_range[0]) * (
            self.ellipticity_range[1] - self.ellipticity_range[0]
        )

        return {
            "sigma": sigma.detach(),
            "ellipticity": ellipticity.detach(),
        }


class BaseTransform:
    def __init__(
        self,
        channel_means: list[float],
        channel_stds: list[float],
    ) -> None:
        self.channel_stds = channel_stds
        self.channel_means = channel_means

        self.transform = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=self.channel_means, std=self.channel_stds),
            ]
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class MultiCropTransform:
    def __init__(self, cfg: AugmentationConfig) -> None:
        self.cfg = cfg

        common = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply(
                [
                    T.RandomRotation(
                        degrees=[0, 180],
                        interpolation=T.InterpolationMode.BILINEAR,
                        fill=0,
                    )
                ],
                p=1.0,
            ),
            T.RandomApply(
                [
                    T.GaussianBlur(
                        kernel_size=5,
                        sigma=self.cfg.blur_sigma,
                    )
                ],
                p=0.5,
            ),
            T.RandomApply(
                [
                    T.GaussianNoise(
                        mean=0,
                        sigma=self.cfg.noise_sigma,
                    )
                ],
                p=0.5,
            ),
            T.RandomErasing(
                p=0.3,
                scale=(0.02, 0.1),
                ratio=(0.5, 2.0),
            ),
        ]

        self.global_transform = T.Compose(
            [
                T.RandomResizedCrop(
                    self.cfg.global_size, self.cfg.global_scale, ratio=(1, 1)
                ),
                *common,
            ]
        )

        self.local_transform = T.Compose(
            [
                T.RandomResizedCrop(
                    self.cfg.local_size, self.cfg.local_scale, ratio=(1, 1)
                ),
                *common,
            ]
        )

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]:
        views = [self.global_transform(x) for _ in range(self.cfg.n_global)]
        views += [self.local_transform(x) for _ in range(self.cfg.n_local)]
        return views


def multicrop_collate_img(batch: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    n_views = len(batch[0])
    return [torch.stack([sample[v] for sample in batch]) for v in range(n_views)]
