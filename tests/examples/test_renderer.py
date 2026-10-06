from __future__ import annotations

import unittest

import numpy as np
import torch

from rebar_inverse import Camera


class RendererTest(unittest.TestCase):
    def test_radius_gradient_increases_coverage(self):
        camera = Camera(
            np.array([[70, 0, 32], [0, 70, 32], [0, 0, 1]]),
            np.eye(4), 64, 64,
        )
        geometry = torch.tensor(
            [[-.3, 0, 2, .3, 0, 2, .05]], device='cuda',
            requires_grad=True,
        )
        self.renderer(camera, [1], 128).render(geometry, 11).sum().backward()
        self.assertGreater(float(geometry.grad[0, 6]), 1)

    def test_rotated_translated_pose_projects_correctly(self):
        pose = np.array([[0, -1, 0, 2], [1, 0, 0, 3],
                         [0, 0, 1, 1], [0, 0, 0, 1]], dtype=float)
        camera = Camera(
            np.array([[70, 0, 32], [0, 70, 32], [0, 0, 1]]),
            pose, 64, 64,
        )
        local = np.array([[-.3, .1, 2], [.3, .1, 2]])
        world = local @ pose[:3, :3].T + pose[:3, 3]
        geometry = torch.tensor(np.r_[world.ravel(), .05][None],
                                dtype=torch.float32, device='cuda')
        image = self.renderer(camera, [1]).render(geometry, 4)
        weight = image[:, :, 0].detach().cpu().numpy()
        y, x = np.indices(weight.shape)
        centroid = [(weight * (x + .5)).sum() / weight.sum(),
                    (weight * (y + .5)).sum() / weight.sum()]
        np.testing.assert_allclose(centroid, [32, 35.5], atol=.3)

    def test_slender_cylinder_has_finite_coverage(self):
        camera = Camera(
            np.array([[70, 0, 32], [0, 70, 32], [0, 0, 1]]),
            np.eye(4), 64, 64,
        )
        geometry = torch.tensor(
            [[-3, 0, 2, 3, 0, 2, .003]], device='cuda',
        )
        rendered = self.renderer(camera, [1]).render(geometry, 9)
        self.assertTrue(torch.isfinite(rendered).all())

    def test_black_occluder_in_another_channel_batch(self):
        camera = Camera(
            np.array([[70, 0, 32], [0, 70, 32], [0, 0, 1]]),
            np.eye(4), 64, 64,
        )
        geometry = torch.tensor([
            [-.5, 0, 2, .5, 0, 2, .10],
            [-.5, .5, 2, .5, .5, 2, .05],
            [-.5, -.5, 2, .5, -.5, 2, .05],
            [-.75, 0, 3, .75, 0, 3, .08],
        ], device='cuda')
        rendered = self.renderer(camera, [1, 2, 3, 4]).render(geometry, 9)
        self.assertLess(float(rendered[:, :, 3].sum()), 1)

    def renderer(self, camera, ids, spp=64):
        try:
            from rebar_inverse.rendering import MitsubaMaskRenderer
        except ImportError as error:
            self.fail(f'Differentiable renderer missing: {error}')
        return MitsubaMaskRenderer(camera, ids, spp)

    def test_offcenter_anisotropic_camera_projection(self):
        camera = Camera(
            np.array([[95, 0, 26], [0, 70, 37], [0, 0, 1]]),
            np.eye(4), 64, 64,
        )
        geometry = torch.tensor(
            [[-0.3, 0.1, 2, 0.3, 0.1, 2, 0.035]], device='cuda',
        )
        rendered = self.renderer(camera, [8]).render(geometry, 4)
        weight = rendered[:, :, 0].detach().cpu().numpy()
        y, x = np.indices(weight.shape)
        centroid = np.array([(weight * (x + .5)).sum(),
                             (weight * (y + .5)).sum()]) / weight.sum()
        expected = camera.project(np.array([[0, .1, 2]]))[0]
        np.testing.assert_allclose(centroid, expected, atol=.35)

    def test_occluded_instance_remains_in_scene(self):
        camera = Camera(
            np.array([[70, 0, 32], [0, 70, 32], [0, 0, 1]]),
            np.eye(4), 64, 64,
        )
        geometry = torch.tensor([
            [-.5, 0, 2, .5, 0, 2, .10],
            [-.75, 0, 3, .75, 0, 3, .08],
        ], device='cuda')
        rendered = self.renderer(camera, [3, 7]).render(geometry, 2)
        self.assertGreater(float(rendered[:, :, 0].sum()), 20)
        self.assertLess(float(rendered[:, :, 1].sum()), 1)

    def test_silhouette_gradient_reaches_torch(self):
        camera = Camera(
            np.array([[70, 0, 32], [0, 70, 32], [0, 0, 1]]),
            np.eye(4), 64, 64,
        )
        geometry = torch.tensor(
            [[-.3, 0, 2, .3, 0, 2, .05]], device='cuda',
            requires_grad=True,
        )
        rendered = self.renderer(camera, [1], 128).render(geometry, 3)
        x = torch.arange(64, device='cuda')[None, :, None] / 64
        (rendered * x).sum().backward()
        self.assertTrue(torch.isfinite(geometry.grad).all())
        self.assertGreater(float(geometry.grad[0, [0, 3]].sum()), .1)


if __name__ == '__main__':
    unittest.main()
