from __future__ import annotations

import unittest

import cv2
import numpy as np

from rebar_inverse import Camera
from rebar_inverse import Observation
from rebar_inverse import RebarReconstructor
from rebar_inverse import ReconstructionConfig


class MultiviewTest(unittest.TestCase):
    def observations(self, scale=1.0):
        observations = []
        for position in (0, .5):
            pose = np.eye(4)
            pose[0, 3] = position * scale
            camera = Camera(
                np.array([[120, 0, 64.5], [0, 120, 48.5], [0, 0, 1]]),
                pose, 128, 96,
            )
            mask = np.zeros((96, 128), dtype=np.int32)
            rods = [
                [[-.8, -.5, 3], [.8, -.5, 3]],
                [[-.8, .5, 3], [.8, .5, 3]],
                [[-.4, -.8, 3], [-.4, .8, 3]],
                [[.4, -.8, 3], [.4, .8, 3]],
            ]
            for index, rod in enumerate(rods):
                points = camera.project(np.array(rod) * scale)
                pixels = np.rint(points - .5).astype(int)
                cv2.line(mask, pixels[0], pixels[1], index + 1, 3)
            observations.append(Observation(mask, camera))
        return observations

    def test_multiview_triangulation_depth(self):
        state = RebarReconstructor(ReconstructionConfig(
            device='cpu')).initialize(self.observations())
        geometry = state.model.geometry().detach().numpy()
        np.testing.assert_allclose(geometry[:, [2, 5]], 6.0, atol=.25)
        self.assertEqual(state.gauge['mode'], 'unit_baseline')

    def test_gauge_is_invariant_to_camera_translation_units(self):
        api = RebarReconstructor(ReconstructionConfig(device='cpu'))
        original = api.initialize(self.observations()).model.geometry()
        scaled = api.initialize(self.observations(7)).model.geometry()
        np.testing.assert_allclose(original.detach().numpy(),
                                   scaled.detach().numpy(), atol=1e-6)

    def test_explicit_mesh_labels_are_recorded(self):
        state = RebarReconstructor(ReconstructionConfig(
            device='cpu')).initialize(self.observations(),
                                     {1: 10, 2: 10, 3: 20, 4: 20})
        self.assertEqual(state.gauge['mesh_labels'], [10, 20])
        self.assertEqual(len(state.model.plane_geometry()), 2)


if __name__ == '__main__':
    unittest.main()
