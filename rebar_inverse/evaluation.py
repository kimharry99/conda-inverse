"""Axis-position errors after independent positive mesh scales."""
from __future__ import annotations

import copy

import numpy as np


class MeshScaleEvaluator:
    """Fit one positive scalar per predicted mesh using matched centers.

    Inputs must share their origin and axis basis. GT is in meters; predictions
    are in arbitrary units. Translation, rotation and per-rod scales are fixed.
    Lengths, radii and directions do not participate in fitting the scale.
    """

    PARALLEL_SINE_TOLERANCE = 1e-10

    def evaluate(self, predicted: list[dict], truth: list[dict],
                 visible_ids: list[int]) -> dict:
        """Align visible cylinders and return per-mesh and per-rod errors.

        Args:
            predicted: Cylinders with instance_id, mesh_id, mid, dir, length,
                radius, expressed in a common origin/axis basis with GT.
            truth: Metric GT cylinders; mesh_id is unnecessary.
            visible_ids: Positive mask IDs, required in both cylinder lists.

        Returns:
            Scales, linear matrices, aligned geometry, and metrics in mm/deg.

        Raises:
            ValueError: Invalid geometry/IDs, missing matches, degenerate
                centers, or a nonpositive fitted scale.
        """
        predictions = self._index(predicted, True)
        targets = self._index(truth, False)
        ids = sorted({self._integer(value, 'visible ID', 1)
                      for value in visible_ids})
        if not ids:
            raise ValueError('At least one visible instance is required.')
        for label, lookup in [('prediction', predictions), ('GT', targets)]:
            missing = sorted(set(ids) - lookup.keys())
            if missing:
                raise ValueError(f'Missing visible {label} IDs: {missing}')
        meshes, rows, aligned = [], [], []
        mesh_ids = sorted({predictions[i]['mesh_id'] for i in ids})
        for mesh_id in mesh_ids:
            members = [i for i in ids if predictions[i]['mesh_id'] == mesh_id]
            centers = np.array([predictions[i]['mid'] for i in members])
            gt_centers = np.array([targets[i]['mid'] for i in members])
            denominator = float(np.sum(centers * centers))
            if denominator <= np.finfo(float).tiny:
                raise ValueError(f'Degenerate center fit for mesh {mesh_id}.')
            scale = float(np.sum(centers * gt_centers) / denominator)
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f'Mesh {mesh_id} needs a positive scale.')
            mesh_rows = []
            for instance_id in members:
                rod = copy.deepcopy(predictions[instance_id])
                rod['mid'] = (np.array(rod['mid']) * scale).tolist()
                rod['length'] *= scale
                rod['radius'] *= scale
                rod['diameter_mm'] = 2000 * rod['radius']
                aligned.append(rod)
                row = self._errors(rod, targets[instance_id])
                row['scale'] = scale
                mesh_rows.append(row)
            rows.extend(mesh_rows)
            meshes.append({
                'mesh_id': mesh_id, 'instance_ids': members, 'scale': scale,
                'scale_units': 'meters per reconstruction unit',
                'linear_matrix': (np.eye(3) * scale).tolist(),
                'metrics': self._aggregate(mesh_rows),
            })
        return {
            'schema_version': 2,
            'method': {
                'transform': 'x_aligned = s_mesh * x_predicted; s_mesh > 0',
                'grouping': 'predicted mesh_id, unchanged',
                'fit': 'equal-weight least squares of matched visible centers',
                'fit_formula': 'sum(dot(p_i, g_i)) / sum(dot(p_i, p_i))',
                'fitted_parameters_per_mesh': 1,
                'translation_fitted': False, 'rotation_fitted': False,
                'lengths_radii_directions_used_for_fit': False,
                'gt_used_for_posthoc_alignment': True,
                'origin': 'shared prediction/GT origin',
                'gt_units': 'meters', 'error_units': 'millimeters/degrees',
                'primary_metric': 'perpendicular_axis_rmse_mm',
                'perpendicular_axis': (
                    'norm(delta - dot(delta, unit_gt_dir) * unit_gt_dir); '
                    'delta = aligned_pred_mid - gt_mid'),
                'along_axis_center': 'abs(dot(delta, unit_gt_dir))',
                'line_to_line': (
                    'abs(dot(delta, cross(unit_pred_dir, unit_gt_dir))) / '
                    'norm(cross); near-parallel fallback: perpendicular_axis'),
                'line_parallel_sine_tolerance': self.PARALLEL_SINE_TOLERANCE,
                'line_extent': 'infinite axes, no endpoint clipping',
                'axis_position_limitations': [
                    'Intersecting infinite axes have zero line distance '
                    'even when angle/finite-rod placement is wrong.',
                    'Perpendicular distance anchors the predicted midpoint '
                    'to the GT axis; it is asymmetric when axes differ.',
                    'Center-based scale fitting remains sensitive to '
                    'longitudinal midpoint errors.',
                ],
                'diagnostic_metrics': [
                    'along_axis_center_rmse_mm', 'center_rmse_mm',
                    'endpoint_rmse_mm', 'length_mae_mm'],
                'axis': 'acos(abs(dot(unit_pred_dir, unit_gt_dir)))',
                'endpoints': 'full finite rods; minimum over endpoint order',
                'overall_weighting': 'equal weight per visible instance',
            },
            'visibility': {
                'evaluated_ids': ids,
                'excluded_gt_ids': sorted(targets.keys() - set(ids)),
                'excluded_prediction_ids': sorted(
                    predictions.keys() - set(ids)),
            },
            'meshes': meshes, 'overall': self._aggregate(rows),
            'per_instance': sorted(rows, key=lambda row: row['instance_id']),
            'aligned_cylinders': sorted(
                aligned, key=lambda rod: rod['instance_id']),
        }

    @staticmethod
    def _integer(value: object, label: str, minimum: int) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{label} must be an integer >= {minimum}.')
        return int(value)

    def _index(self, cylinders: list[dict], require_mesh: bool) -> dict:
        indexed = {}
        for original in cylinders:
            rod = copy.deepcopy(original)
            instance_id = self._integer(rod['instance_id'], 'instance ID', 1)
            if instance_id in indexed:
                raise ValueError(f'Duplicate instance ID: {instance_id}')
            rod['instance_id'] = instance_id
            if require_mesh:
                rod['mesh_id'] = self._integer(rod['mesh_id'], 'mesh ID', 0)
            for key in ('mid', 'dir'):
                vector = np.asarray(rod[key], dtype=float)
                if vector.shape != (3,) or not np.isfinite(vector).all():
                    raise ValueError(f'Invalid {key} for ID {instance_id}.')
                if key == 'dir':
                    norm = np.linalg.norm(vector)
                    if norm <= np.finfo(float).tiny:
                        raise ValueError('Cylinder direction cannot be zero.')
                    vector = vector / norm
                rod[key] = vector.tolist()
            for key in ('length', 'radius'):
                value = float(rod[key])
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f'{key} must be positive and finite.')
                rod[key] = value
            indexed[instance_id] = rod
        return indexed

    @classmethod
    def _errors(cls, predicted: dict, truth: dict) -> dict:
        center, gt_center = np.array(predicted['mid']), np.array(truth['mid'])
        direction, gt_direction = (np.array(predicted['dir']),
                                   np.array(truth['dir']))
        dot = float(np.dot(direction, gt_direction))
        if dot < 0:
            gt_direction = -gt_direction
        delta = center - gt_center
        along = float(np.dot(delta, gt_direction))
        perpendicular = float(np.linalg.norm(delta - along * gt_direction))
        cross = np.cross(direction, gt_direction)
        sine = float(np.linalg.norm(cross))
        near_parallel = sine <= cls.PARALLEL_SINE_TOLERANCE
        line_distance = (perpendicular if near_parallel else
                         abs(float(np.dot(delta, cross / sine))))
        offsets = np.array([[-.5], [.5]])
        endpoints = center + offsets * predicted['length'] * direction
        gt_endpoints = gt_center + offsets * truth['length'] * gt_direction
        endpoint_mm = np.linalg.norm(endpoints - gt_endpoints, axis=1) * 1000
        row = {
            'instance_id': predicted['instance_id'],
            'mesh_id': predicted['mesh_id'],
            'perpendicular_axis_error_mm': perpendicular * 1000,
            'line_to_line_distance_mm': line_distance * 1000,
            'along_axis_center_error_mm': abs(along) * 1000,
            'axes_near_parallel': near_parallel,
            'center_error_mm': float(np.linalg.norm(delta) * 1000),
            'endpoint_0_error_mm': float(endpoint_mm[0]),
            'endpoint_1_error_mm': float(endpoint_mm[1]),
            'endpoint_squared_mean_mm2': float(np.mean(endpoint_mm ** 2)),
            'axis_error_deg': float(np.degrees(np.arccos(
                np.clip(abs(dot), 0, 1)))),
        }
        for key, factor in [('length', 1000), ('radius', 1000),
                            ('diameter', 2000)]:
            source = 'radius' if key == 'diameter' else key
            row[f'{key}_signed_error_mm'] = (
                predicted[source] - truth[source]) * factor
            row[f'predicted_{key}_mm'] = predicted[source] * factor
            row[f'gt_{key}_mm'] = truth[source] * factor
        return row

    @staticmethod
    def _aggregate(rows: list[dict]) -> dict:
        centers = np.array([row['center_error_mm'] for row in rows])
        angles = np.array([row['axis_error_deg'] for row in rows])
        metrics = {
            'count': len(rows),
            'center_rmse_mm': float(np.sqrt(np.mean(centers ** 2))),
            'center_mean_mm': float(np.mean(centers)),
            'center_median_mm': float(np.median(centers)),
            'endpoint_rmse_mm': float(np.sqrt(np.mean([
                row['endpoint_squared_mean_mm2'] for row in rows]))),
            'axis_mean_deg': float(np.mean(angles)),
            'axis_median_deg': float(np.median(angles)),
        }
        for name, column in [
                ('perpendicular_axis', 'perpendicular_axis_error_mm'),
                ('line_to_line', 'line_to_line_distance_mm'),
                ('along_axis_center', 'along_axis_center_error_mm')]:
            distances = np.array([row[column] for row in rows])
            metrics[f'{name}_rmse_mm'] = float(np.sqrt(
                np.mean(distances ** 2)))
            metrics[f'{name}_mean_mm'] = float(np.mean(distances))
            metrics[f'{name}_median_mm'] = float(np.median(distances))
        for parameter in ('length', 'radius', 'diameter'):
            errors = np.array([row[f'{parameter}_signed_error_mm']
                               for row in rows])
            metrics[f'{parameter}_mae_mm'] = float(np.mean(np.abs(errors)))
            metrics[f'{parameter}_rmse_mm'] = float(np.sqrt(
                np.mean(errors ** 2)))
            metrics[f'{parameter}_bias_mm'] = float(np.mean(errors))
        return metrics
