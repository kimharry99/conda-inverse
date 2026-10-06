"""GT topology labels must be stable under camera and ID ordering changes."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from rebar_inverse.gt_labels import GroundTruthMeshLabeler


class GroundTruthMeshLabelerTest(unittest.TestCase):
    @staticmethod
    def rods():
        return [{'instance_id': i + 1, 'name': 'unrelated-name',
                 'mid': [0, 0, z], 'dir': direction, 'radius': .005}
                for i, (z, direction) in enumerate([
                    (0, [1, 0, 0]), (.01, [0, 1, 0]),
                    (.07, [1, 0, 0]), (.08, [0, 1, 0])])]

    def test_touching_planes_form_two_meshes_without_name_assumptions(self):
        self.assertEqual(GroundTruthMeshLabeler().assign(self.rods()),
                         {1: 0, 2: 0, 3: 1, 4: 1})

    def test_labels_are_pose_scale_and_order_invariant(self):
        rods = copy.deepcopy(self.rods())
        rotation = np.array([[1, 0, 0], [0, .6, -.8], [0, .8, .6]])
        for rod in rods:
            rod['mid'] = (rotation @ np.array(rod['mid']) * 7
                          + [1, 2, 3]).tolist()
            rod['dir'] = (-rotation @ np.array(rod['dir'])).tolist()
            rod['radius'] *= 7
        self.assertEqual(GroundTruthMeshLabeler().assign(rods[::-1]),
                         {1: 0, 2: 0, 3: 1, 4: 1})

    def test_ambiguous_or_invalid_gt_is_rejected(self):
        rods = self.rods()
        cases = [[], [rods[0], rods[0]],
                 [{**rod, 'dir': [1, 0, 0]} for rod in rods],
                 [{**rod, 'radius': 0} for rod in rods],
                 [{**rod, 'radius': .1} for rod in rods]]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                GroundTruthMeshLabeler().assign(case)


if __name__ == '__main__':
    unittest.main()
