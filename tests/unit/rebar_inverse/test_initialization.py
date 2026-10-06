"""Plane directions should resist a few corrupted mask centerlines."""
from __future__ import annotations

import unittest

import numpy as np

from rebar_inverse import Camera
from rebar_inverse import ReconstructionConfig
from rebar_inverse.initialization import MaskInitializer


class _TestableMaskInitializer(MaskInitializer):
    def plane_normal(self, records, camera, fallback=None):
        return self._plane_normal(records, camera, fallback)


class PlaneNormalTest(unittest.TestCase):
    def test_missing_line_family_keeps_shared_mask_normal(self):
        camera = Camera(np.eye(3), np.eye(4), 8, 8)
        normal = np.array([0, .8, .6])
        initializer = _TestableMaskInitializer(
            ReconstructionConfig(device='cpu'))
        inferred = initializer.plane_normal({}, camera, normal)
        np.testing.assert_allclose(inferred, normal)

    def test_mesh_label_number_does_not_determine_depth_order(self):
        from rebar_inverse import Observation
        from rebar_inverse import RebarReconstructor
        mask = np.zeros((128, 128), dtype=np.int32)
        mask[34:37, 10:118] = 1
        mask[10:118, 34:37] = 2
        mask[82:89, 10:118] = 3
        mask[10:118, 82:89] = 4
        camera = Camera(
            np.array([[150, 0, 64], [0, 150, 64], [0, 0, 1]]),
            np.eye(4), 128, 128)
        state = RebarReconstructor(ReconstructionConfig(
            device='cpu')).initialize([Observation(mask, camera)],
                                     {1: 0, 2: 0, 3: 1, 4: 1})
        planes = state.model.plane_geometry()
        self.assertLess(planes[1]['origin'][2], planes[0]['origin'][2])

    def test_parallel_families_with_outliers_recover_plane(self):
        camera = Camera(np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
                        np.eye(4), 640, 480)
        normal = np.array([0, .8, .6])
        u = np.cross(normal, [.2, .3, 1])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        records = {}
        for direction, other in [(u, v), (v, u)]:
            for offset in np.linspace(-1, 1, 8):
                center = np.array([0, 0, 4]) + offset * other
                ends = camera.project(center + np.array([[-1], [1]])
                                      * direction)
                line = np.cross(np.r_[ends[0], 1], np.r_[ends[1], 1])
                line /= np.linalg.norm(line[:2])
                records[len(records)] = [(camera, ends, 1, line)]
        # Real records hold an Observation rather than a bare camera.
        from rebar_inverse import Observation
        observation = Observation(np.zeros((480, 640), dtype=np.int32), camera)
        for key, entries in records.items():
            _, ends, width, line = entries[0]
            records[key] = [(observation, ends, width, line)]
        for line in ([.2, 1, -60], [1, -.3, -70]):
            records[len(records)] = [(observation, None, 1, np.array(line))]
        inferred = _TestableMaskInitializer(
            ReconstructionConfig(device='cpu')).plane_normal(records, camera)
        self.assertGreater(abs(float(inferred @ normal)), .99999)


if __name__ == '__main__':
    unittest.main()
