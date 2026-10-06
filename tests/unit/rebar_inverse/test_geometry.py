from __future__ import annotations

import unittest

import numpy as np
import torch


class GeometryTest(unittest.TestCase):
    def api(self):
        try:
            from rebar_inverse import Camera, Observation
            from rebar_inverse.initialization import MaskInitializer
            from rebar_inverse.types import ReconstructionConfig
        except ImportError as error:
            self.fail(f'Reconstruction API is missing: {error}')
        return Camera, Observation, MaskInitializer, ReconstructionConfig

    def test_resize_preserves_pixel_center_projection(self):
        camera_type, _, _, _ = self.api()
        camera = camera_type(
            np.array([[80, 0, 31], [0, 90, 23], [0, 0, 1]]),
            np.eye(4), 64, 48,
        )
        point = np.array([[0.2, -0.1, 2.0]])
        np.testing.assert_allclose(
            camera.resized(32, 24).project(point),
            camera.project(point) * 0.5,
        )

    def test_invalid_pose_is_rejected(self):
        camera_type, _, _, _ = self.api()
        pose = np.eye(4)
        pose[0, 0] = 2
        with self.assertRaises(ValueError):
            camera_type(np.eye(3), pose, 32, 32)

    def test_fractional_instance_ids_are_rejected(self):
        camera_type, observation_type, _, _ = self.api()
        camera = camera_type(np.eye(3), np.eye(4), 8, 8)
        with self.assertRaises(ValueError):
            observation_type(np.ones((8, 8)) * 0.5, camera)

    def test_subpixel_rod_survives_coarse_target(self):
        camera_type, observation_type, _, _ = self.api()
        mask = np.zeros((32, 32), dtype=np.int32)
        mask[:, 1] = 7
        camera = camera_type(np.eye(3), np.eye(4), 32, 32)
        target = observation_type(mask, camera).coverage([7], 8, 8)
        self.assertAlmostEqual(float(target.sum()), 8 * 0.25)

    def test_planarity_survives_parameter_updates(self):
        camera_type, observation_type, initializer_type, config = self.api()
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[12:15, 5:59] = 1
        mask[38:41, 5:59] = 2
        mask[5:59, 29:32] = 3
        camera = camera_type(
            np.array([[70, 0, 32], [0, 70, 32], [0, 0, 1]]),
            np.eye(4), 64, 64,
        )
        state = initializer_type(config(device='cpu')).initialize(
            [observation_type(mask, camera)]
        )
        model = state.model
        with torch.no_grad():
            model.plane_tilts.add_(0.1)
            model.centers.add_(0.07)
            model.angles.add_(0.2)
        geometry = model.geometry().detach().numpy()
        planes = model.plane_geometry()
        normal = planes[0]['normal']
        origin = planes[0]['origin']
        for endpoint in (geometry[:, :3], geometry[:, 3:6]):
            np.testing.assert_allclose(
                (endpoint - origin) @ normal, 0, atol=2e-7
            )
        self.assertTrue(np.all(geometry[:, 6] > 0))
        self.assertEqual(model.instance_ids, [1, 2, 3])

    def test_empty_observations_are_rejected(self):
        _, _, initializer_type, config = self.api()
        with self.assertRaises(ValueError):
            initializer_type(config(device='cpu')).initialize([])


if __name__ == '__main__':
    unittest.main()
