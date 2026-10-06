"""Known-answer checks for per-mesh scale-only GT evaluation."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from rebar_inverse.evaluation import MeshScaleEvaluator


class MeshScaleEvaluatorTest(unittest.TestCase):
    """Scale fitting must not absorb translation or shape errors."""

    @staticmethod
    def cylinder(instance_id, mesh_id, mid, length=1.0, radius=0.01):
        return {
            'instance_id': instance_id, 'mesh_id': mesh_id,
            'mid': mid, 'dir': [1.0, 0.0, 0.0],
            'length': length, 'radius': radius,
        }

    def test_independent_scales_and_unsigned_axes(self):
        predicted = [self.cylinder(1, 0, [1, 0, 2]),
                     self.cylinder(2, 1, [0, 1, 3])]
        before = copy.deepcopy(predicted)
        truth = copy.deepcopy(predicted)
        for rod, scale in zip(truth, [2, 5]):
            rod['mid'] = (np.array(rod['mid']) * scale).tolist()
            rod['length'] *= scale
            rod['radius'] *= scale
            rod['dir'] = [-1, 0, 0]
        result = MeshScaleEvaluator().evaluate(predicted, truth, [1, 2])
        self.assertEqual(predicted, before)
        self.assertEqual([m['scale'] for m in result['meshes']], [2, 5])
        self.assertEqual(result['overall']['center_rmse_mm'], 0)
        self.assertEqual(result['overall']['endpoint_rmse_mm'], 0)
        self.assertEqual(result['overall']['axis_mean_deg'], 0)
        self.assertEqual(result['overall']['diameter_mae_mm'], 0)
        self.assertEqual(result['aligned_cylinders'][1]['length'], 5)

    def test_center_fit_does_not_fit_translation_or_dimensions(self):
        predicted = [self.cylinder(1, 0, [1, 0, 2]),
                     self.cylinder(2, 0, [-1, 0, 2])]
        truth = copy.deepcopy(predicted)
        for rod in truth:
            rod['mid'][1] += 1
            rod['length'] = 2
            rod['radius'] = 0.03
        result = MeshScaleEvaluator().evaluate(predicted, truth, [1, 2])
        self.assertEqual(result['meshes'][0]['scale'], 1)
        metrics = result['overall']
        self.assertAlmostEqual(metrics['center_rmse_mm'], 1000)
        self.assertAlmostEqual(metrics['endpoint_rmse_mm'], 1000 * 1.25**.5)
        self.assertAlmostEqual(metrics['length_mae_mm'], 1000)
        self.assertAlmostEqual(metrics['diameter_mae_mm'], 40)
        np.testing.assert_allclose(result['meshes'][0]['linear_matrix'],
                                   np.eye(3))

    def test_visibility_controls_fit_and_instance_matching(self):
        predicted = [self.cylinder(2, 0, [0, 0, 1]),
                     self.cylinder(1, 0, [0, 0, 1])]
        truth = [self.cylinder(1, 0, [0, 0, 2]),
                 self.cylinder(2, 0, [0, 0, 100]),
                 self.cylinder(3, 0, [0, 0, 300])]
        result = MeshScaleEvaluator().evaluate(predicted, truth, [1])
        self.assertEqual(result['meshes'][0]['scale'], 2)
        self.assertEqual(result['overall']['count'], 1)
        self.assertEqual(result['visibility']['excluded_gt_ids'], [2, 3])

    def test_missing_visible_prediction_or_gt_fails(self):
        rod = self.cylinder(1, 0, [0, 0, 1])
        for predicted, truth in [([], [rod]), ([rod], [])]:
            with self.subTest(predicted=predicted), self.assertRaises(
                    ValueError):
                MeshScaleEvaluator().evaluate(predicted, truth, [1])

    def test_degenerate_negative_and_invalid_inputs_fail(self):
        rod = self.cylinder(1, 0, [0, 0, 1])
        for field, value in [('mid', [0, 0, 0]), ('mid', [0, 0, -1]),
                             ('mid', [0, np.nan, 1]), ('dir', [0, 0, 0]),
                             ('radius', -1)]:
            bad = copy.deepcopy(rod)
            bad[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    MeshScaleEvaluator().evaluate([bad], [rod], [1])
        with self.assertRaises(ValueError):
            MeshScaleEvaluator().evaluate([rod, rod], [rod], [1])
        with self.assertRaises(ValueError):
            MeshScaleEvaluator().evaluate([rod], [rod], [])


    def test_longitudinal_truncation_preserves_axis_position(self):
        predicted = self.cylinder(1, 0, [0, 0, 2], length=.1)
        truth = self.cylinder(1, 0, [3, 0, 2], length=7)
        truth['dir'] = [-3, 0, 0]
        result = MeshScaleEvaluator().evaluate([predicted], [truth], [1])
        row = result['per_instance'][0]
        self.assertEqual(result['meshes'][0]['scale'], 1)
        self.assertEqual(row['perpendicular_axis_error_mm'], 0)
        self.assertEqual(row['line_to_line_distance_mm'], 0)
        self.assertEqual(row['along_axis_center_error_mm'], 3000)
        self.assertEqual(row['center_error_mm'], 3000)
        self.assertEqual(result['method']['primary_metric'],
                         'perpendicular_axis_rmse_mm')

    def test_axis_distance_geometry_cases(self):
        cases = [('parallel', [2, 0, 0], 4000),
                 ('antiparallel', [-2, 0, 0], 4000),
                 ('skew', [0, 0, 2], 4000),
                 ('intersecting', [0, 2, 0], 0)]
        for name, direction, expected in cases:
            with self.subTest(name=name):
                predicted = self.cylinder(1, 0, [0, 0, 2])
                predicted['dir'] = direction
                truth = self.cylinder(1, 0, [3, 4, 2])
                truth['dir'] = [-5, 0, 0]
                result = MeshScaleEvaluator().evaluate(
                    [predicted], [truth], [1])
                row = result['per_instance'][0]
                self.assertEqual(row['line_to_line_distance_mm'], expected)
                self.assertEqual(row['perpendicular_axis_error_mm'], 4000)
                self.assertEqual(row['along_axis_center_error_mm'], 3000)
                self.assertAlmostEqual(row['center_error_mm'] ** 2,
                                       4000 ** 2 + 3000 ** 2)

    def test_near_parallel_tolerance_boundary(self):
        tolerance = 1e-10
        for name, sine, expected in [
                ('below', np.nextafter(tolerance, 0), 1000),
                ('at', tolerance, 1000),
                ('above', np.nextafter(tolerance, np.inf), 0)]:
            with self.subTest(name=name):
                predicted = self.cylinder(1, 0, [0, 0, 2])
                predicted['dir'] = [1, sine, 0]
                truth = self.cylinder(1, 0, [0, 1, 2])
                result = MeshScaleEvaluator().evaluate(
                    [predicted], [truth], [1])
                row = result['per_instance'][0]
                self.assertEqual(row['line_to_line_distance_mm'], expected)
                self.assertEqual(row['axes_near_parallel'], sine <= tolerance)
                self.assertEqual(
                    result['method']['line_parallel_sine_tolerance'],
                    tolerance)

    def test_axis_aggregation_weights_visible_rods_equally(self):
        predicted = [self.cylinder(1, 0, [0, 0, 1]),
                     self.cylinder(2, 0, [0, 0, 1]),
                     self.cylinder(3, 1, [0, 0, 1])]
        truth = [self.cylinder(1, 0, [.01, .003, 2]),
                 self.cylinder(2, 0, [.02, .004, 2]),
                 self.cylinder(3, 1, [.03, .012, 5])]
        result = MeshScaleEvaluator().evaluate(predicted, truth, [3, 1, 2])
        self.assertEqual([m['scale'] for m in result['meshes']], [2, 5])
        overall = result['overall']
        for prefix in ('perpendicular_axis', 'line_to_line'):
            self.assertAlmostEqual(overall[f'{prefix}_rmse_mm'],
                                   (169 / 3) ** .5)
            self.assertAlmostEqual(overall[f'{prefix}_mean_mm'], 19 / 3)
            self.assertEqual(overall[f'{prefix}_median_mm'], 4)
            self.assertAlmostEqual(
                result['meshes'][0]['metrics'][f'{prefix}_rmse_mm'],
                (25 / 2) ** .5)
        self.assertAlmostEqual(overall['along_axis_center_rmse_mm'],
                               (1400 / 3) ** .5)
        self.assertAlmostEqual(
            overall['center_rmse_mm'] ** 2,
            overall['perpendicular_axis_rmse_mm'] ** 2
            + overall['along_axis_center_rmse_mm'] ** 2)


if __name__ == '__main__':
    unittest.main()
