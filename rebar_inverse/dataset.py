"""Read-only v2_2layer adapter with explicit camera-frame behavior."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from rebar_inverse.types import Camera
from rebar_inverse.types import Observation
from rebar_inverse.gt_labels import GroundTruthMeshLabeler


class V2Dataset:
    """Load masks/K only; location/look_at never imply an unstated roll."""

    def __init__(self, root: str | Path = 'data/rebar/v2_2layer'):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

    def frame_path(self, relative: str) -> Path:
        """Resolve a scene/frame path inside the immutable dataset."""
        path = (self.root / relative).resolve()
        if self.root not in path.parents or len(Path(relative).parts) != 2:
            raise ValueError('Use a dataset-relative scene/frame path.')
        return path

    def load_frame(self, relative: str,
                   camera_to_world: np.ndarray | None = None) -> Observation:
        """Load one frame; default identity pose denotes its camera frame."""
        frame = self.frame_path(relative)
        metadata = json.loads((frame / 'meta.json').read_text())
        mask = np.load(frame / 'instance_id.npy', allow_pickle=False)
        width = metadata['camera']['width']
        height = metadata['camera']['height']
        k = np.array(metadata['intrinsics'], dtype=float)
        k[0] *= width
        k[1] *= height
        pose = np.eye(4) if camera_to_world is None else camera_to_world
        semantic_path = frame / 'semantic.png'
        if semantic_path.exists():
            semantic = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
            if semantic is None:
                raise ValueError(f'Cannot decode {semantic_path}')
            if semantic.ndim == 3:
                semantic = semantic.max(axis=2)
            if not np.array_equal(semantic > 0, mask > 0):
                raise ValueError('Semantic and instance foreground disagree.')
        return Observation(mask, Camera(k, pose, width, height), relative)

    def load_manifest(self, manifest: str | Path,
                      poses: dict[str, np.ndarray] | None = None) -> list:
        """Load one scene; multiview requires explicit poses."""
        frames = Path(manifest).read_text().splitlines()
        frames = [line.strip() for line in frames if line.strip()]
        if not frames or len(frames) != len(set(frames)):
            raise ValueError('Manifest must contain unique scene/frame rows.')
        if len({Path(f).parts[0] for f in frames}) != 1:
            raise ValueError('A reconstruction manifest must use one scene.')
        if len(frames) > 1 and (poses is None or set(frames) != set(poses)):
            raise ValueError('Multiview v2 requires verified explicit poses '
                             'for every manifest row; no inferred up/roll.')
        return [self.load_frame(f, None if poses is None else poses[f])
                for f in frames]

    def provenance(self, manifest: str | Path) -> dict:
        """Hash selection and source files without modifying the originals."""
        manifest = Path(manifest)
        frames = [s.strip() for s in manifest.read_text().splitlines()
                  if s.strip()]
        files = {}
        for relative in frames:
            frame = self.frame_path(relative)
            for filename in ('instance_id.npy', 'meta.json'):
                files[f'{relative}/{filename}'] = hashlib.sha256(
                    (frame / filename).read_bytes()).hexdigest()
        return {
            'name': 'v2_2layer', 'root': str(self.root), 'frames': frames,
            'sample_manifest_sha256': hashlib.sha256(
                manifest.read_bytes()).hexdigest(), 'input_sha256': files,
            'integrity_manifest':
                'data/rebar/manifests/v2_2layer.sha256',
            'gt_coordinates': 'camera frame, meters; not used for fitting',
        }

    def gt_mesh_labels(self, observations: list[Observation]) -> tuple:
        """Load GT-derived membership for visible IDs and its provenance.

        Uses full GT geometry only to derive labels. Requires consistent
        topology across the supplied views of a single scene.
        """
        names = [view.name for view in observations]
        if (not names or any(len(Path(name).parts) != 2 for name in names)
                or len(names) != len(set(names))
                or len({Path(name).parts[0] for name in names}) != 1):
            raise ValueError('GT labels require unique views of one scene.')
        assignment, hashes = None, {}
        for name in names:
            path = self.frame_path(name) / 'gt_cylinders.json'
            raw = path.read_bytes()
            current = GroundTruthMeshLabeler().assign(json.loads(raw))
            if assignment is not None and current != assignment:
                raise ValueError('GT mesh memberships disagree across frames.')
            assignment = current
            hashes[f'{name}/gt_cylinders.json'] = (
                hashlib.sha256(raw).hexdigest())
        visible = sorted({int(i) for view in observations
                          for i in np.unique(view.mask) if i > 0})
        if not visible or not set(visible).issubset(assignment):
            raise ValueError('GT labels must cover every visible instance.')
        labels = {i: assignment[i] for i in visible}
        return labels, {
            'source': 'gt', 'method': 'parallel contact-plane components',
            'contact_tolerance_m': 1e-5,
            'label_order': 'minimum instance ID per GT component',
            'gt_fields_for_label_derivation': ['mid', 'dir', 'radius'],
            'gt_values_used_as_initial_parameters': False,
            'input_sha256': hashes,
            'instance_to_mesh': {str(i): label for i, label in labels.items()},
        }
