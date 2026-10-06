"""Cylinder axes parameterized directly on shared planes."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from rebar_inverse.directions import DirectionFamilies


class PlanarCylinderModel(nn.Module):
    """Each mesh has one plane; cylinders have only in-plane coordinates."""

    def __init__(self, instance_ids: list[int], mesh_indices: np.ndarray,
                 origins: np.ndarray, bases: np.ndarray,
                 endpoints: np.ndarray, radii: np.ndarray,
                 device: str, anchor_first: bool,
                 direction_families: np.ndarray | None = None) -> None:
        super().__init__()
        self.instance_ids = instance_ids
        self.register_buffer('mesh_indices', torch.as_tensor(
            mesh_indices, dtype=torch.long, device=device))
        self.register_buffer('origins', self._tensor(origins, device))
        self.register_buffer('bases', self._tensor(bases, device))
        weights = np.ones(len(origins), dtype=np.float32)
        if anchor_first:
            weights[0] = 0
        self.register_buffer('offset_weights', self._tensor(weights, device))
        local = endpoints - origins[mesh_indices, None, :]
        uv = np.einsum('nkj,nij->nki', local, bases[mesh_indices, :2])
        delta = uv[:, 1] - uv[:, 0]
        self.centers = nn.Parameter(self._tensor(uv.mean(axis=1), device))
        angles = np.arctan2(delta[:, 1], delta[:, 0])
        if direction_families is None:
            families = np.full(len(angles), -1, dtype=np.int64)
            angle_indices = np.arange(len(angles))
        else:
            families = np.asarray(direction_families)
            if (families.shape != mesh_indices.shape
                    or not np.issubdtype(families.dtype, np.integer)
                    or np.any(families < 0)):
                raise ValueError('Direction families require one label per rod.')
            _, angle_indices = np.unique(
                np.column_stack((mesh_indices, families)),
                axis=0, return_inverse=True)
            angles = np.array([
                DirectionFamilies.median_angle(angles[angle_indices == group])
                for group in range(int(angle_indices.max()) + 1)])
        self.register_buffer('direction_families', torch.as_tensor(
            families, dtype=torch.long, device=device))
        self.register_buffer('angle_indices', torch.as_tensor(
            angle_indices, dtype=torch.long, device=device))
        self.angles = nn.Parameter(self._tensor(angles, device))
        self.log_lengths = nn.Parameter(self._tensor(
            np.log(np.maximum(np.linalg.norm(delta, axis=1), 1e-4)),
            device))
        self.log_radii = nn.Parameter(self._tensor(np.log(radii), device))
        self.plane_tilts = nn.Parameter(torch.zeros(
            (len(origins), 2), device=device))
        self.plane_offsets = nn.Parameter(torch.zeros(
            len(origins), device=device))

    @staticmethod
    def _tensor(value: np.ndarray, device: str) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    def frames(self) -> tuple[torch.Tensor, ...]:
        """Return differentiable plane origin, in-plane basis and normal."""
        u0, v0, n0 = self.bases.unbind(dim=1)
        n = functional.normalize(
            n0 + self.plane_tilts[:, :1] * u0
            + self.plane_tilts[:, 1:] * v0, dim=1)
        u = functional.normalize(u0 - (u0 * n).sum(1, True) * n, dim=1)
        v = torch.linalg.cross(n, u)
        origin = self.origins + (
            self.plane_offsets * self.offset_weights)[:, None] * n0
        return origin, u, v, n

    def geometry(self) -> torch.Tensor:
        """Return N x 7 rows [start xyz, end xyz, radius]."""
        origin, u, v, _ = self.frames()
        index = self.mesh_indices
        center = origin[index] + self.centers[:, :1] * u[index]
        center = center + self.centers[:, 1:] * v[index]
        angles = self.angles[self.angle_indices]
        direction = (angles.cos()[:, None] * u[index]
                     + angles.sin()[:, None] * v[index])
        half = self.log_lengths.exp()[:, None] * direction * 0.5
        return torch.cat((center - half, center + half,
                          self.log_radii.exp()[:, None]), dim=1)

    def plane_geometry(self) -> list[dict]:
        """Export shared plane equations in reconstruction coordinates."""
        origin, _, _, normal = self.frames()
        return [
            {'mesh_id': i, 'origin': o.tolist(), 'normal': n.tolist()}
            for i, (o, n) in enumerate(zip(
                origin.detach().cpu().numpy(),
                normal.detach().cpu().numpy()))
        ]

    def cylinders(self) -> list[dict]:
        """Export positive-radius cylinders and their mesh membership."""
        geometry = self.geometry().detach().cpu().numpy()
        meshes = self.mesh_indices.cpu().numpy()
        families = self.direction_families.cpu().numpy()
        output = []
        for instance_id, mesh, family, row in zip(
                self.instance_ids, meshes, families, geometry):
            delta = row[3:6] - row[:3]
            length = float(np.linalg.norm(delta))
            output.append({
                'instance_id': instance_id, 'mesh_id': int(mesh),
                'mid': ((row[:3] + row[3:6]) * 0.5).tolist(),
                'dir': (delta / length).tolist(), 'length': length,
                'radius': float(row[6]),
            })
            if family >= 0:
                output[-1]['direction_family'] = int(family)
        return output
