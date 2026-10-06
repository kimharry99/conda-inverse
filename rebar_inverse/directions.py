"""Mask-derived axial direction families, without an orthogonality prior."""
from __future__ import annotations

import numpy as np


class DirectionFamilies:
    """Cluster unoriented in-plane axes independently in each mesh."""

    @staticmethod
    def infer(angles: np.ndarray, meshes: np.ndarray) -> np.ndarray:
        """Return at most two local family labels per mesh.

        Doubled angles identify reversed axes. Family names are arbitrary;
        neither clustering nor shared-angle fitting prescribes their angle.
        Membership stays fixed during subsequent continuous optimization.
        """
        labels = np.zeros(len(angles), dtype=np.int64)
        for mesh in np.unique(meshes):
            rows = np.flatnonzero(meshes == mesh)
            features = np.column_stack((np.cos(2 * angles[rows]),
                                        np.sin(2 * angles[rows])))
            distances = ((features[:, None] - features[None]) ** 2).sum(2)
            first, second = np.unravel_index(
                np.argmax(distances), distances.shape)
            if distances[first, second] < 1e-12:
                continue
            centers = features[[first, second]].copy()
            assigned = np.zeros(len(rows), dtype=np.int64)
            for _ in range(32):
                updated = ((features[:, None] - centers[None]) ** 2).sum(
                    2).argmin(1)
                if np.array_equal(updated, assigned):
                    break
                assigned = updated
                for family in range(2):
                    centers[family] = features[assigned == family].mean(0)
            labels[rows] = assigned
        return labels

    @staticmethod
    def median_angle(angles: np.ndarray) -> float:
        """Fit an axial median, robust to isolated partial-mask directions."""
        residuals = .5 * np.angle(np.exp(
            2j * (angles[:, None] - angles[None])))
        reference = angles[np.abs(residuals).sum(0).argmin()]
        residual = .5 * np.angle(np.exp(2j * (angles - reference)))
        return float(reference + np.median(residual))
