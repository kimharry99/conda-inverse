"""Validated inputs and settings for scale-ambiguous reconstruction."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Camera:
    """Pinhole camera: pixel K, camera-to-world, +X right/+Y down/+Z front.

    Pixel centers are (column + 0.5, row + 0.5). Distortion and skew must
    be removed before use. Translation units may be arbitrary but must
    agree across views.
    """

    intrinsic: np.ndarray
    camera_to_world: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        k = np.array(self.intrinsic, dtype=np.float64, copy=True)
        pose = np.array(self.camera_to_world, dtype=np.float64, copy=True)
        if self.width < 1 or self.height < 1:
            raise ValueError('Camera dimensions must be positive.')
        if k.shape != (3, 3) or not np.isfinite(k).all():
            raise ValueError('intrinsic must be a finite 3x3 matrix.')
        if k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError('Focal lengths must be positive.')
        if not np.allclose(k[2], [0, 0, 1]):
            raise ValueError('intrinsic last row must be [0, 0, 1].')
        if abs(k[0, 1]) + abs(k[1, 0]) > 1e-8:
            raise ValueError('Rectify camera skew before reconstruction.')
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError('camera_to_world must be a finite 4x4 matrix.')
        rotation = pose[:3, :3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
                or not np.allclose(pose[3], [0, 0, 0, 1])):
            raise ValueError('camera_to_world must be a rigid SE(3) pose.')
        k.setflags(write=False)
        pose.setflags(write=False)
        object.__setattr__(self, 'intrinsic', k)
        object.__setattr__(self, 'camera_to_world', pose)

    def resized(self, width: int, height: int) -> Camera:
        """Return intrinsics in the resized pixel-edge convention."""
        k = self.intrinsic.copy()
        k[0] *= width / self.width
        k[1] *= height / self.height
        return Camera(k, self.camera_to_world, width, height)

    def rays(self, pixels: np.ndarray) -> np.ndarray:
        """Return world-space ray directions for N pixel coordinates."""
        homogeneous = np.column_stack([pixels, np.ones(len(pixels))])
        local = homogeneous @ np.linalg.inv(self.intrinsic).T
        return local @ self.camera_to_world[:3, :3].T

    def project(self, points: np.ndarray) -> np.ndarray:
        """Project world points; reject points behind the pinhole."""
        local = (points - self.camera_to_world[:3, 3])
        local = local @ self.camera_to_world[:3, :3]
        if np.any(local[:, 2] <= 0):
            raise ValueError('Cannot project points behind the camera.')
        image = local @ self.intrinsic.T
        return image[:, :2] / image[:, 2:3]


@dataclass(frozen=True)
class Observation:
    """Visible instance raster; 0 is background, IDs persist across views."""

    mask: np.ndarray
    camera: Camera
    name: str = ''

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask)
        if mask.shape != (self.camera.height, self.camera.width):
            raise ValueError('Mask shape must match camera dimensions.')
        if (not np.issubdtype(mask.dtype, np.integer)
                or np.any(mask < 0)):
            raise ValueError('Mask must contain nonnegative integer IDs.')
        mask = mask.astype(np.int64, copy=True)
        mask.setflags(write=False)
        object.__setattr__(self, 'mask', mask)

    def coverage(self, ids: list[int], width: int,
                 height: int) -> np.ndarray:
        """Area-average each binary mask, preserving thin coarse rods."""
        return np.stack([
            cv2.resize(
                (self.mask == instance_id).astype(np.float32),
                (width, height), interpolation=cv2.INTER_AREA,
            ) for instance_id in ids
        ], axis=-1)


@dataclass(frozen=True)
class ReconstructionConfig:
    """Optional resolution schedule, independent of data-usage stages."""

    max_sizes: tuple[int, ...] = (128, 256, 512)
    iterations: tuple[int, ...] = (30, 30, 40)
    learning_rate: float = 0.001
    spp: int = 64
    seed: int = 7
    num_planes: int = 1
    plane_separation: float = 0.04
    regularization: float = 0.001
    device: str = 'cuda'
    share_directions: bool = True

    def __post_init__(self) -> None:
        if (not self.max_sizes or len(self.max_sizes) != len(self.iterations)
                or any(s < 8 for s in self.max_sizes)
                or any(i < 1 for i in self.iterations)
                or any(a >= b for a, b in zip(
                    self.max_sizes, self.max_sizes[1:]))):
            raise ValueError('Use increasing sizes >= 8 and positive steps.')
        if (self.num_planes < 1 or self.spp < 1 or self.seed < 0
                or not np.isfinite(self.learning_rate)
                or self.learning_rate <= 0
                or not np.isfinite(self.plane_separation)
                or self.plane_separation <= 0
                or not np.isfinite(self.regularization)
                or self.regularization < 0):
            raise ValueError('Invalid reconstruction configuration.')
