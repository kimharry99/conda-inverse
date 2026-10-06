"""GT membership is read independently of mask-only geometric initialization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rebar_inverse import Camera
from rebar_inverse import Observation
from rebar_inverse.dataset import V2Dataset


class GroundTruthAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.observations = []
        self.paths = []
        for frame in range(2):
            name = f'scene/frame_{frame:04d}'
            path = self.root / name / 'gt_cylinders.json'
            path.parent.mkdir(parents=True)
            rods = [{'instance_id': i + 1, 'mid': [0, 0, z + frame],
                     'dir': [1, 0, 0] if i % 2 else [0, 1, 0],
                     'radius': .005}
                    for i, z in enumerate([0, .01, .07, .08])]
            path.write_text(json.dumps(rods))
            self.paths.append(path)
            mask = np.full((4, 4), 1 if frame == 0 else 3, dtype=np.int32)
            self.observations.append(Observation(
                mask, Camera(np.eye(3), np.eye(4), 4, 4), name))

    def test_labels_use_full_gt_then_select_union_of_visible_ids(self):
        before = [path.read_bytes() for path in self.paths]
        labels, provenance = V2Dataset(self.root).gt_mesh_labels(
            self.observations)
        self.assertEqual(labels, {1: 0, 3: 1})
        self.assertEqual(provenance['source'], 'gt')
        self.assertFalse(provenance['gt_values_used_as_initial_parameters'])
        self.assertEqual(before, [path.read_bytes() for path in self.paths])
        self.assertEqual(list(provenance['input_sha256'].values()),
                         [hashlib.sha256(raw).hexdigest() for raw in before])

    def test_inconsistent_frame_topology_is_rejected(self):
        rods = json.loads(self.paths[1].read_text())
        rods[0]['instance_id'], rods[2]['instance_id'] = 3, 1
        self.paths[1].write_text(json.dumps(rods))
        with self.assertRaisesRegex(ValueError, 'disagree'):
            V2Dataset(self.root).gt_mesh_labels(self.observations)

    def test_missing_visible_gt_id_is_rejected(self):
        observation = self.observations[0]
        observation = Observation(
            np.full((4, 4), 99, dtype=np.int32), observation.camera,
            observation.name)
        with self.assertRaisesRegex(ValueError, 'every visible'):
            V2Dataset(self.root).gt_mesh_labels([observation])


if __name__ == '__main__':
    unittest.main()
