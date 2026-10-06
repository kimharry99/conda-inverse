"""File-level evaluation with a known, nonidentity camera pose."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rebar_inverse.evaluate import V2Evaluation


class V2EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / 'data'
        self.frame = self.dataset / 'scene' / 'frame_0000'
        self.frame.mkdir(parents=True)
        np.save(self.frame / 'instance_id.npy', np.ones((4, 4), dtype=np.int32))
        truth = [{'instance_id': 1, 'mid': [0, 0, 2], 'dir': [1, 0, 0],
                  'length': 1, 'radius': .01}]
        (self.frame / 'gt_cylinders.json').write_text(json.dumps(truth))
        self.document = {
            'cylinders': [{'instance_id': 1, 'mesh_id': 0, 'mid': [1, 2, 4],
                           'dir': [0, 1, 0], 'length': .5, 'radius': .005}],
            'planes': [], 'warnings': [], 'config': {'seed': 7},
            'provenance': {
                'dataset': {'frames': ['scene/frame_0000']},
                'normalized_cameras': [{
                    'name': 'scene/frame_0000', 'width': 4, 'height': 4,
                    'intrinsic': np.eye(3).tolist(),
                    'camera_to_world': [[0, -1, 0, 1], [1, 0, 0, 2],
                                        [0, 0, 1, 3], [0, 0, 0, 1]],
                }],
            },
        }
        self.result = self.root / 'result.json'
        self.result.write_text(json.dumps(self.document))

    def test_known_pose_conversion_and_readonly_report(self):
        files = [self.result, self.frame / 'instance_id.npy',
                 self.frame / 'gt_cylinders.json']
        before = [hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in files]
        evaluator = V2Evaluation()
        report = evaluator.evaluate(self.result, self.dataset)
        self.assertEqual(report['meshes'][0]['scale'], 2)
        self.assertEqual(report['overall']['endpoint_rmse_mm'], 0)
        np.testing.assert_allclose(
            report['aligned_cylinders'][0]['mid'], [0, 0, 2])
        path = evaluator.save(report, self.root / 'output')
        self.assertEqual(json.loads(path.read_text())['overall']['count'], 1)
        self.assertEqual(len((path.parent / 'per_instance.csv')
                             .read_text().splitlines()), 2)
        self.assertEqual(before, [hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in files])
        with self.assertRaises(ValueError):
            evaluator.save(report, self.dataset / 'forbidden')

    def test_missing_camera_and_changed_mask_are_rejected(self):
        self.document['provenance']['dataset']['input_sha256'] = {
            'scene/frame_0000/instance_id.npy': 'wrong-checksum'}
        self.result.write_text(json.dumps(self.document))
        with self.assertRaises(ValueError):
            V2Evaluation().evaluate(self.result, self.dataset)

        self.document['provenance']['dataset']['input_sha256'] = {}
        self.document['provenance']['normalized_cameras'] = []
        self.result.write_text(json.dumps(self.document))
        with self.assertRaises(ValueError):
            V2Evaluation().evaluate(self.result, self.dataset)

    def test_gt_plane_diagnostics_do_not_change_prediction_membership(self):
        truth, predicted = [], []
        for index in range(8):
            rod = {'instance_id': index + 1, 'mid': [0, 0, 2 + index // 2],
                   'dir': [1, 0, 0] if index % 2 else [0, 1, 0],
                   'length': 1, 'radius': .01}
            truth.append(rod)
            predicted.append({**rod, 'mesh_id': index % 2})
        self.document['cylinders'] = predicted
        camera = self.document['provenance']['normalized_cameras'][0]
        camera['camera_to_world'] = np.eye(4).tolist()
        self.result.write_text(json.dumps(self.document))
        (self.frame / 'gt_cylinders.json').write_text(json.dumps(truth))
        np.save(self.frame / 'instance_id.npy',
                np.tile(np.arange(1, 9, dtype=np.int32), 2).reshape(4, 4))
        report = V2Evaluation().evaluate(self.result, self.dataset)
        diagnostic = report['gt_plane_diagnostics']
        self.assertEqual(len(diagnostic['center_planes']), 4)
        self.assertEqual(diagnostic['predicted_mesh_plane_counts']['0'],
                         {'0': 1, '1': 1, '2': 1, '3': 1})
        self.assertEqual(report['meshes'][0]['instance_ids'], [1, 3, 5, 7])


    def test_axis_metrics_survive_report_serialization(self):
        truth_path = self.frame / 'gt_cylinders.json'
        truth = json.loads(truth_path.read_text())
        truth[0]['mid'] = [3, 4, 2]
        truth_path.write_text(json.dumps(truth))
        evaluator = V2Evaluation()
        report = evaluator.evaluate(self.result, self.dataset)
        path = evaluator.save(report, self.root / 'axis_output')
        saved = json.loads(path.read_text())
        self.assertEqual(saved['schema_version'], 2)
        self.assertEqual(saved['overall']['perpendicular_axis_rmse_mm'], 4000)
        self.assertEqual(saved['overall']['line_to_line_rmse_mm'], 4000)
        with (path.parent / 'per_instance.csv').open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(float(row['perpendicular_axis_error_mm']), 4000)
        self.assertEqual(float(row['line_to_line_distance_mm']), 4000)
        markdown = (path.parent / 'report.md').read_text()
        self.assertIn('수직 축 RMSE (mm)', markdown)
        self.assertIn('Line-to-line RMSE (mm)', markdown)
        self.assertIn('축 방향 중심 RMSE (mm)', markdown)


if __name__ == '__main__':
    unittest.main()
