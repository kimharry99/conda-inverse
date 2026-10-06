"""Mask-only line extraction, triangulation and planar initialization."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rebar_inverse.directions import DirectionFamilies
from rebar_inverse.model import PlanarCylinderModel
from rebar_inverse.types import Camera
from rebar_inverse.types import Observation
from rebar_inverse.types import ReconstructionConfig


@dataclass
class Initialization:
    """Model plus cameras in a consistent arbitrary-unit gauge."""

    model: PlanarCylinderModel
    observations: list[Observation]
    gauge: dict
    warnings: list[str]


class MaskInitializer:
    """Estimate axes without accessing RGB, depth or GT cylinders."""

    def __init__(self, config: ReconstructionConfig) -> None:
        self.config = config

    def initialize(self, observations: list[Observation],
                   mesh_ids: dict[int, int] | None = None) -> Initialization:
        """Build a model; optional mesh_ids disambiguate mesh membership.

        Single-view depth is a prior, not an observable. Multiple views
        must share instance IDs and consistent explicit poses.
        """
        if not observations:
            raise ValueError('At least one observation is required.')
        observations, gauge = self._normalize_cameras(observations)
        ids = sorted(set(int(i) for view in observations
                         for i in np.unique(view.mask) if i > 0))
        if not ids:
            raise ValueError('No foreground instances found.')
        if mesh_ids is not None and set(mesh_ids) != set(ids):
            raise ValueError('mesh_ids must map every visible instance ID.')
        records = {i: [] for i in ids}
        for view in observations:
            for instance_id in ids:
                ys, xs = np.nonzero(view.mask == instance_id)
                if len(xs) < 3:
                    continue
                pixels = np.column_stack((xs + 0.5, ys + 0.5))
                mean = pixels.mean(axis=0)
                _, _, vh = np.linalg.svd(pixels - mean, full_matrices=False)
                direction = vh[0]
                along = (pixels - mean) @ direction
                ends = mean + np.array([
                    along.min() - 0.5, along.max() + 0.5,
                ])[:, None] * direction
                across = (pixels - mean) @ vh[1]
                width = max(float(np.quantile(across, 0.95)
                                  - np.quantile(across, 0.05) + 1), 1)
                line = np.cross(np.r_[ends[0], 1], np.r_[ends[1], 1])
                line /= np.linalg.norm(line[:2])
                records[instance_id].append((view, ends, width, line))
        missing = [i for i in ids if not records[i]]
        if missing:
            raise ValueError(f'Need >= 3 pixels in one view for IDs {missing}')
        first_camera = observations[0].camera
        normal = self._plane_normal(records, first_camera)
        origin = (first_camera.camera_to_world[:3, 3]
                  + first_camera.camera_to_world[:3, 2])
        endpoints, radii, triangulated = [], [], []
        warnings = []
        for instance_id in ids:
            entries = sorted(records[instance_id],
                             key=lambda item: -np.linalg.norm(
                                 item[1][1] - item[1][0]))
            records[instance_id] = entries
            view, pixels, width, _ = entries[0]
            world_ends = self._triangulate(entries)
            triangulated.append(world_ends is not None)
            if world_ends is None:
                rays = view.camera.rays(pixels)
                center = view.camera.camera_to_world[:3, 3]
                denominator = rays @ normal
                distance = (origin - center) @ normal / denominator
                if not np.isfinite(distance).all() or np.any(distance <= 0):
                    raise ValueError('Plane prior intersects behind camera; '
                                     'provide overlapping views or rectify.')
                world_ends = center + rays * distance[:, None]
            endpoints.append(world_ends)
            camera_point = (world_ends.mean(axis=0)
                            - view.camera.camera_to_world[:3, 3])
            depth = camera_point @ view.camera.camera_to_world[:3, 2]
            focal = np.sqrt(np.prod(view.camera.intrinsic.diagonal()[:2]))
            radii.append(max(width * depth / (2 * focal), 1e-5))
        endpoints = np.asarray(endpoints)
        radii = np.asarray(radii)
        if not all(triangulated):
            warnings.append('Some axes use a single-view plane/depth prior; '
                            'hidden extent and layer depth are ambiguous.')
        if mesh_ids is None:
            count = min(self.config.num_planes, len(ids))
            if count > 1:
                # Equal-diameter prior: larger corrected apparent radii
                # suggest the nearer layer. This is deliberately a heuristic.
                order = np.argsort(-radii, kind='stable')
                indices = np.empty(len(ids), dtype=int)
                for group, subset in enumerate(np.array_split(order, count)):
                    indices[subset] = group
                warnings.append('Mesh IDs inferred from apparent width using '
                                'an equal-diameter prior; pass mesh_ids for '
                                'known topology. Membership is fixed.')
            else:
                indices = np.zeros(len(ids), dtype=int)
        else:
            labels = sorted(set(mesh_ids.values()))
            indices = np.array([labels.index(mesh_ids[i]) for i in ids])
            gauge['mesh_labels'] = labels
            count = len(labels)
        depth_ranks = np.arange(count)
        if mesh_ids is not None and count > 1 and not all(triangulated):
            apparent_radii = np.array([
                np.median(radii[indices == group]) for group in range(count)])
            depth_ranks = np.argsort(np.argsort(-apparent_radii, kind='stable'))
            warnings.append('Initial mesh depth order uses median apparent '
                            'radii and an equal-diameter prior; label numbers '
                            'do not encode depth.')
        gauge['initial_mesh_depth_ranks'] = depth_ranks.tolist()
        origins, bases = [], []
        for group in range(count):
            selected = indices == group
            group_normal = normal
            if mesh_ids is not None and np.count_nonzero(selected) >= 4:
                group_normal = self._plane_normal(
                    {ids[row]: records[ids[row]]
                     for row in np.flatnonzero(selected)}, first_camera, normal)
            group_origin = origin + group_normal * (
                (depth_ranks[group] - depth_ranks[0])
                * self.config.plane_separation)
            group_basis = self._basis(group_normal)
            supported = selected & np.asarray(triangulated)
            if np.any(supported):
                points = endpoints[supported].reshape(-1, 3)
                group_origin = points.mean(axis=0)
                _, singular, vh = np.linalg.svd(points - group_origin)
                if (len(singular) > 1
                        and singular[1] > singular[0] * 1e-3):
                    group_basis = self._basis(vh[-1])
            # Other bars on this plane supply depth when one axis is
            # parallel to the stereo baseline and cannot be triangulated.
            for row in np.flatnonzero(selected & ~supported):
                view = records[ids[row]][0][0]
                center = view.camera.camera_to_world[:3, 3]
                rays = endpoints[row] - center
                scale = ((group_origin - center) @ group_basis[2]
                         / (rays @ group_basis[2]))
                if not np.isfinite(scale).all() or np.any(scale <= 0):
                    raise ValueError('Inferred plane lies behind a camera.')
                endpoints[row] = center + rays * scale[:, None]
                radii[row] *= float(np.mean(scale))
            origins.append(group_origin)
            bases.append(group_basis)
        families = None
        if self.config.share_directions:
            delta = endpoints[:, 1] - endpoints[:, 0]
            uv = np.einsum('nj,nij->ni', delta, np.array(bases)[indices, :2])
            families = DirectionFamilies.infer(
                np.arctan2(uv[:, 1], uv[:, 0]), indices)
        gauge['direction_constraint'] = {
            'enabled': self.config.share_directions,
            'family_source': ('mask_initialized_axes'
                              if families is not None else None),
            'orthogonality_constraint': False,
            'membership_fixed_during_optimization': True,
        }
        model = PlanarCylinderModel(
            ids, indices, np.array(origins), np.array(bases), endpoints,
            radii, self.config.device, gauge['mode'] == 'unit_depth',
            direction_families=families,
        )
        return Initialization(model, observations, gauge, warnings)

    @staticmethod
    def _normalize_cameras(observations: list[Observation]) -> tuple:
        origin = observations[0].camera.camera_to_world[:3, 3]
        baseline = max(np.linalg.norm(v.camera.camera_to_world[:3, 3]
                                      - origin) for v in observations)
        scale = baseline if baseline > 1e-8 else 1.0
        normalized = []
        for view in observations:
            pose = view.camera.camera_to_world.copy()
            pose[:3, 3] = (pose[:3, 3] - origin) / scale
            camera = Camera(view.camera.intrinsic, pose,
                            view.camera.width, view.camera.height)
            normalized.append(Observation(view.mask, camera, view.name))
        return normalized, {
            'mode': 'unit_baseline' if baseline > 1e-8 else 'unit_depth',
            'units': 'arbitrary', 'input_origin': origin.tolist(),
            'input_units_per_output_unit': (
                float(scale) if baseline > 1e-8 else None),
            'metric_scale_recovered': False,
            'coordinates': 'input world axes; camera +X right +Y down +Z front',
        }

    @staticmethod
    def _basis(normal: np.ndarray) -> np.ndarray:
        normal = normal / np.linalg.norm(normal)
        reference = np.eye(3)[np.argmin(np.abs(normal))]
        u = np.cross(reference, normal)
        u /= np.linalg.norm(u)
        return np.stack([u, np.cross(normal, u), normal])

    @staticmethod
    def _plane_normal(records: dict, camera: Camera,
                      fallback_normal: np.ndarray | None = None) -> np.ndarray:
        """Fit two consensus vanishing directions, rejecting corrupted lines."""
        lines = [entry[3] for entries in records.values() for entry in entries
                 if entry[0].camera is camera]
        fallback = (camera.camera_to_world[:3, 2] if fallback_normal is None
                    else fallback_normal)
        if len(lines) < 4:
            return fallback
        normals = np.asarray(lines) @ camera.intrinsic
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        remaining = np.ones(len(normals), dtype=bool)
        directions = []
        for _ in range(2):
            indices = np.flatnonzero(remaining)
            if len(indices) < 2:
                return fallback
            a, b = np.triu_indices(len(indices), 1)
            # Bound memory for large grids with deterministic selection.
            if len(a) > 4096:
                chosen = np.linspace(0, len(a) - 1, 4096, dtype=int)
                a, b = a[chosen], b[chosen]
            candidates = np.cross(normals[indices[a]], normals[indices[b]])
            norms = np.linalg.norm(candidates, axis=1)
            valid = norms > 1e-7
            if not np.any(valid):
                return fallback
            candidates = candidates[valid] / norms[valid, None]
            residuals = np.abs(normals @ candidates.T)
            support = (residuals < .002) & remaining[:, None]
            counts = support.sum(axis=0)
            cost = np.where(support, residuals ** 2, 0).sum(axis=0)
            best = np.lexsort((cost, -counts))[0]
            inliers = support[:, best]
            if np.count_nonzero(inliers) < 2:
                return fallback
            _, _, vh = np.linalg.svd(normals[inliers], full_matrices=True)
            directions.append(vh[-1])
            remaining[inliers] = False
        normal = np.cross(*directions)
        if np.linalg.norm(normal) < 0.2:
            return fallback
        normal /= np.linalg.norm(normal)
        normal *= 1 if normal[2] >= 0 else -1
        if normal[2] < 0.25:
            return fallback
        return camera.camera_to_world[:3, :3] @ normal

    @staticmethod
    def _triangulate(entries: list) -> np.ndarray | None:
        if len(entries) < 2:
            return None
        normals, offsets = [], []
        for view, _, _, line in entries:
            camera = view.camera
            normal = camera.camera_to_world[:3, :3] @ (
                camera.intrinsic.T @ line)
            normal /= np.linalg.norm(normal)
            normals.append(normal)
            offsets.append(normal @ camera.camera_to_world[:3, 3])
        _, singular, vh = np.linalg.svd(normals, full_matrices=True)
        if singular[1] < 1e-3:
            return None
        direction = vh[-1]
        point = np.linalg.lstsq(normals, offsets, rcond=None)[0]
        positions = []
        for view, pixels, _, _ in entries:
            center = view.camera.camera_to_world[:3, 3]
            for ray in view.camera.rays(pixels):
                along, distance = np.linalg.lstsq(
                    np.column_stack([direction, -ray]), center - point,
                    rcond=None)[0]
                if distance <= 0:
                    return None
                positions.append(along)
        return point + np.array([min(positions), max(positions)])[:, None] * (
            direction)
