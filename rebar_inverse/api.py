"""Public reconstruction API and reproducible result artifacts."""
from __future__ import annotations

import copy
from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable

import drjit as dr
import mitsuba as mi
import numpy as np
import torch

from rebar_inverse.initialization import Initialization
from rebar_inverse.initialization import MaskInitializer
from rebar_inverse.rendering import MitsubaMaskRenderer
from rebar_inverse.types import Observation
from rebar_inverse.types import ReconstructionConfig


@dataclass
class ReconstructionResult:
    """Geometry, diagnostics and the explicit arbitrary-unit scale gauge."""

    cylinders: list[dict]
    initial_cylinders: list[dict]
    planes: list[dict]
    gauge: dict
    metrics: dict
    history: list[dict]
    warnings: list[str]
    config: dict
    provenance: dict
    predictions: dict[str, np.ndarray]

    def save(self, directory: str | Path) -> Path:
        """Write JSON geometry/trace and NPZ prediction masks separately."""
        directory = Path(directory).resolve()
        protected = Path('data/rebar').resolve()
        if directory == protected or protected in directory.parents:
            raise ValueError('Outputs must be outside the original dataset.')
        directory.mkdir(parents=True, exist_ok=True)
        document = {
            'schema_version': 1, 'cylinders': self.cylinders,
            'initial_cylinders': self.initial_cylinders,
            'planes': self.planes, 'gauge': self.gauge,
            'metrics': self.metrics, 'history': self.history,
            'warnings': self.warnings, 'config': self.config,
            'provenance': self.provenance,
        }
        path = directory / 'result.json'
        path.write_text(json.dumps(document, indent=2, allow_nan=False),
                        encoding='utf-8')
        np.savez_compressed(directory / 'predictions.npz', **self.predictions)
        return path


class RebarReconstructor:
    """Initialize mask-derived planar cylinders and optimize on CUDA."""

    def __init__(self, config: ReconstructionConfig | None = None) -> None:
        self.config = config if config is not None else ReconstructionConfig()

    def initialize(self, observations: list[Observation],
                   mesh_ids: dict[int, int] | None = None) -> Initialization:
        """Return initial model and normalized cameras without rendering."""
        return MaskInitializer(self.config).initialize(observations, mesh_ids)

    def reconstruct(self, observations: list[Observation],
                    mesh_ids: dict[int, int] | None = None,
                    progress: Callable[[dict], None] | None = None,
                    provenance: dict | None = None) -> ReconstructionResult:
        """Fit masks; IDs must identify the same rods across views.

        Args:
            observations: Integer masks with pixel K and rigid camera poses.
            mesh_ids: Optional instance ID to common-plane label mapping.
            progress: Optional callback after each objective evaluation.
            provenance: Optional dataset/manifest identifiers to save.

        Returns:
            Initial/final cylinders, planes, metrics and scale metadata.

        Raises:
            ValueError: Invalid inputs or a non-CUDA optimization device.
            RuntimeError: CUDA unavailable or nonfinite optimization.
        """
        if self.config.device != 'cuda':
            raise ValueError('Optimization requires device="cuda"; CPU is '
                             'supported only for initialize().')
        state = self.initialize(observations, mesh_ids)
        output_provenance = self._provenance(observations)
        label_source = (provenance or {}).get('mesh_labels', {}).get(
            'source', 'provided' if mesh_ids is not None else 'mask')
        if label_source == 'gt' and mesh_ids is None:
            raise ValueError('GT label provenance requires explicit mesh_ids.')
        output_provenance['gt_used_for_initialization'] = label_source == 'gt'
        output_provenance['gt_geometry_used_as_initial_parameters'] = False
        output_provenance['mesh_label_source'] = label_source
        output_provenance['plane_orientation_estimator'] = (
            'per-mesh calibrated mask-line consensus, threshold=0.002')
        state.gauge['mesh_label_source'] = label_source
        initial_cylinders = state.model.cylinders()
        initial_parameters = copy.deepcopy(state.model.state_dict())
        initial_metrics, initial_predictions = self._evaluate(state)
        history = []
        for stage, (max_size, iterations) in enumerate(zip(
                self.config.max_sizes, self.config.iterations)):
            renderers, targets = self._stage_inputs(state, max_size)
            optimizer = torch.optim.Adam(
                state.model.parameters(),
                lr=self.config.learning_rate / (2 ** stage),
            )
            best_loss = float('inf')
            best_parameters = copy.deepcopy(state.model.state_dict())
            for iteration in range(iterations + 1):
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for index, (renderer, target) in enumerate(zip(
                        renderers, targets)):
                    prediction = renderer.render(
                        state.model.geometry(), self.config.seed + index * 97)
                    loss = self._loss(prediction, target) / len(renderers)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite rendering loss.')
                    total += float(loss.detach())
                    if iteration < iterations:
                        loss.backward()
                regularization = self._regularization(
                    state, initial_parameters)
                total += float(regularization.detach())
                if not np.isfinite(total):
                    raise RuntimeError('Nonfinite regularized objective.')
                if total < best_loss:
                    best_loss = total
                    best_parameters = copy.deepcopy(state.model.state_dict())
                record = {
                    'stage': stage, 'max_size': max_size,
                    'iteration': iteration, 'loss': total,
                }
                history.append(record)
                if progress is not None:
                    progress(record)
                if iteration == iterations:
                    break
                regularization.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    state.model.parameters(), 1.0, error_if_nonfinite=True)
                record['gradient_norm'] = float(gradient_norm)
                optimizer.step()
                self._constrain_parameters(state, initial_parameters)
            state.model.load_state_dict(best_parameters)
            del renderers, targets, optimizer
        final_metrics, predictions = self._evaluate(state)
        if final_metrics['loss'] > initial_metrics['loss']:
            state.model.load_state_dict(initial_parameters)
            final_metrics, predictions = initial_metrics, initial_predictions
            state.warnings.append('Optimization did not improve the final '
                                  'evaluation loss; initial model retained.')
        output_provenance['dataset'] = provenance or {}
        output_provenance['normalized_cameras'] = [
            {'name': v.name, 'intrinsic': v.camera.intrinsic.tolist(),
             'camera_to_world': v.camera.camera_to_world.tolist(),
             'width': v.camera.width, 'height': v.camera.height}
            for v in state.observations
        ]
        output_predictions = {
            **{f'initial_{k}': v for k, v in initial_predictions.items()},
            **{f'final_{k}': v for k, v in predictions.items()},
        }
        return ReconstructionResult(
            state.model.cylinders(), initial_cylinders,
            state.model.plane_geometry(), state.gauge,
            {'initial': initial_metrics, 'final': final_metrics}, history,
            state.warnings, asdict(self.config), output_provenance,
            output_predictions,
        )

    def _constrain_parameters(self, state, initial_parameters):
        """Apply the planar model's existing numerical bounds after Adam."""
        with torch.no_grad():
            for parameter, name in (
                    (state.model.log_lengths, 'log_lengths'),
                    (state.model.log_radii, 'log_radii')):
                parameter.copy_(torch.maximum(torch.minimum(
                    parameter, initial_parameters[name] + 1.6),
                    initial_parameters[name] - 1.6))
            state.model.plane_tilts.clamp_(-1, 1)
            state.model.plane_offsets.clamp_(-.4, .4)

    def _stage_inputs(self, state: Initialization,
                      max_size: int) -> tuple[list, list]:
        renderers, targets = [], []
        for view in state.observations:
            scale = min(1.0, max_size / max(view.mask.shape))
            width = max(1, round(view.camera.width * scale))
            height = max(1, round(view.camera.height * scale))
            renderers.append(MitsubaMaskRenderer(
                view.camera.resized(width, height), state.model.instance_ids,
                self.config.spp))
            targets.append(torch.as_tensor(view.coverage(
                state.model.instance_ids, width, height), device='cuda'))
        return renderers, targets

    @staticmethod
    def _loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        area = target.sum(dim=(0, 1)).clamp_min(1)
        return (((prediction - target) ** 2).sum(dim=(0, 1)) / area).mean()

    def _regularization(self, state: Initialization,
                        initial_parameters: dict) -> torch.Tensor:
        model = state.model
        penalty = ((model.log_lengths - initial_parameters['log_lengths'])
                   ** 2).mean()
        penalty = penalty + ((model.log_radii
                              - initial_parameters['log_radii']) ** 2).mean()
        penalty = penalty + model.plane_offsets.square().mean()
        penalty = penalty + model.plane_tilts.square().mean()
        return self.config.regularization * penalty

    def _evaluate(self, state: Initialization) -> tuple[dict, dict]:
        renderers, targets = self._stage_inputs(
            state, self.config.max_sizes[-1])
        losses, per_view, masks = [], [], {}
        with torch.no_grad():
            for index, (renderer, target) in enumerate(zip(renderers, targets)):
                prediction = renderer.render(
                    state.model.geometry(), self.config.seed + index * 97)
                losses.append(float(self._loss(prediction, target)))
                intersection = torch.minimum(prediction, target).sum((0, 1))
                union = torch.maximum(prediction, target).sum((0, 1))
                visible = target.sum((0, 1)) > 0
                iou = intersection / union.clamp_min(1e-8)
                visible_values = iou[visible].cpu().tolist()
                per_view.append({
                    'name': state.observations[index].name,
                    'mean_iou': (float(np.mean(visible_values))
                                 if visible_values else None),
                    'per_instance_iou': {
                        str(instance_id): float(value)
                        for instance_id, value, is_visible in zip(
                            state.model.instance_ids, iou.cpu(), visible.cpu())
                        if is_visible
                    },
                    'width': renderer.camera.width,
                    'height': renderer.camera.height,
                })
                confidence, labels = prediction.max(dim=2)
                ids = torch.tensor(state.model.instance_ids, device='cuda')
                labels = torch.where(confidence >= .5, ids[labels], 0)
                masks[f'view_{index}'] = labels.cpu().numpy()
        return {
            'loss': float(np.mean(losses)),
            'mean_iou': float(np.mean([
                v['mean_iou'] for v in per_view if v['mean_iou'] is not None
            ])),
            'iou_definition': 'visible-ID mean area-coverage min/max IoU',
            'views': per_view,
        }, masks

    def _provenance(self, observations: list[Observation]) -> dict:
        root = Path(__file__).resolve().parent.parent
        source = hashlib.sha256()
        for path in sorted((root / 'rebar_inverse').glob('*.py')):
            source.update(path.name.encode())
            source.update(path.read_bytes())
        try:
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
                stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        return {
            'code_commit': commit, 'source_sha256': source.hexdigest(),
            'seed': self.config.seed, 'torch': torch.__version__,
            'mitsuba': mi.__version__, 'drjit': dr.__version__,
            'variant': 'cuda_ad_rgb', 'cuda': torch.version.cuda,
            'renderer': 'prb_projective, max_depth=1, implicit cylinders',
            'geometry_representation': 'analytic cylinder + two disk caps',
            'intersection_solver': 'quadratic side roots and plane/disk caps',
            'gpu': torch.cuda.get_device_name(),
            'gt_used_for_initialization': False,
            'observations': [
                {'name': v.name, 'mask_sha256': hashlib.sha256(
                    v.mask.tobytes()).hexdigest()}
                for v in observations
            ],
        }
