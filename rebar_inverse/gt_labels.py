"""Derive v2_2layer mesh membership from metric GT contact planes."""
from __future__ import annotations

import numpy as np


class GroundTruthMeshLabeler:
    """Group parallel GT axis planes whose cylinder thicknesses touch.

    v2_2layer has no mesh_id field. Two crossing rod families in one mesh have
    centers one diameter apart. Their normal intervals overlap, whereas the
    two physical meshes are separated by a gap. Object names are not used.
    Only discrete labels leave this adapter; no metric initialization is made.
    """

    def assign(self, cylinders: list[dict], expected_meshes: int = 2,
               tolerance_m: float = 1e-5) -> dict[int, int]:
        """Return stable labels ordered by the smallest instance ID per mesh.

        Args:
            cylinders: Full GT list, including invisible rods; metric mid,
                direction dir, radius, and integer instance_id are required.
            expected_meshes: Required number of separate contact components.
            tolerance_m: Contact tolerance in meters (default 0.01 mm).

        Returns:
            All GT instance IDs mapped to a zero-based mesh label.

        Raises:
            ValueError: Invalid, nonparallel, ambiguous, or incomplete geometry.
        """
        if (len(cylinders) < 4 or expected_meshes < 1 or tolerance_m < 0
                or not np.isfinite(tolerance_m)):
            raise ValueError('Insufficient GT or invalid labeling settings.')
        ids = [rod['instance_id'] for rod in cylinders]
        if (any(isinstance(i, bool) or not isinstance(i, int) or i <= 0
                for i in ids) or len(set(ids)) != len(ids)):
            raise ValueError('GT IDs must be unique positive integers.')
        centers = np.asarray([rod['mid'] for rod in cylinders], dtype=float)
        directions = np.asarray(
            [rod['dir'] for rod in cylinders], dtype=float)
        radii = np.asarray([rod['radius'] for rod in cylinders], dtype=float)
        if (centers.shape != (len(ids), 3) or directions.shape != centers.shape
                or radii.shape != (len(ids),)
                or not np.isfinite(centers).all()
                or not np.isfinite(directions).all()
                or not np.isfinite(radii).all() or np.any(radii <= 0)):
            raise ValueError('Invalid GT centers, directions, or radii.')
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms < 1e-12):
            raise ValueError('GT directions must be nonzero.')
        directions = directions / norms[:, None]
        _, singular, axes = np.linalg.svd(directions, full_matrices=False)
        if (singular[1] < singular[0] * 1e-6
                or singular[2] > singular[0] * 1e-6):
            raise ValueError('GT must define two parallel mesh planes.')
        offsets = centers @ axes[-1]
        lower, upper = offsets - radii, offsets + radii
        groups, edge = [], -np.inf
        for index in np.argsort(lower, kind='stable'):
            if lower[index] > edge + tolerance_m:
                groups.append([])
            groups[-1].append(ids[index])
            edge = max(edge, upper[index])
        if len(groups) != expected_meshes:
            raise ValueError(f'Expected {expected_meshes} GT meshes, '
                             f'found {len(groups)} contact components.')
        groups.sort(key=min)
        return {instance_id: label for label, group in enumerate(groups)
                for instance_id in sorted(group)}
