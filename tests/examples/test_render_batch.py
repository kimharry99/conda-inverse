"""Regression checks for all-cylinder differentiation in one AD batch."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from rebar_inverse import Camera
from rebar_inverse.rendering import MitsubaMaskRenderer


class RenderBatchTest(unittest.TestCase):
    """Exercise real visibility gradients across the RGB packing boundary."""

    def camera(self) -> Camera:
        return Camera(
            np.array([[40, 0, 32], [0, 40, 24], [0, 0, 1]]),
            np.eye(4), 64, 48,
        )

    def geometry(self, count: int) -> torch.Tensor:
        return torch.tensor([
            [x, -.6, 2, x, .6, 2, .08]
            for x in np.linspace(-1.2, 1.2, count)
        ], dtype=torch.float32, device='cuda', requires_grad=True)

    def test_all_visible_cylinders_receive_radius_gradient(self):
        for count in (2, 3, 4, 7):
            with self.subTest(instance_count=count):
                geometry = self.geometry(count)
                renderer = MitsubaMaskRenderer(
                    self.camera(), list(range(1, count + 1)), 128)
                image = renderer.render(geometry, 7)
                self.assertEqual(image.shape, (48, 64, count))
                image.sum().backward()
                self.assertTrue(torch.isfinite(geometry.grad).all())
                self.assertTrue(
                    torch.all(geometry.grad[:, 6] > 100),
                    f'Missing radius derivatives: {geometry.grad[:, 6]}')

    def test_joint_gradient_matches_channel_gradient_sum(self):
        renderer = MitsubaMaskRenderer(self.camera(), [1, 2, 3, 4], 256)
        geometry = self.geometry(4)
        weights = torch.tensor([1, 2, 3, 4], device='cuda')
        (renderer.render(geometry, 13) * weights).sum().backward()
        joint_gradient = geometry.grad.detach().clone()
        accumulated = torch.zeros_like(geometry)
        for index in range(4):
            independent = geometry.detach().clone().requires_grad_(True)
            image = renderer.render(independent, 13)
            (image[:, :, index].sum() * weights[index]).backward()
            accumulated += independent.grad
        self.assertTrue(torch.all(accumulated[:, 6] > 100))
        np.testing.assert_allclose(
            joint_gradient.cpu().numpy(), accumulated.cpu().numpy(),
            rtol=1e-4, atol=.003,
        )

    def test_permuted_instances_keep_radius_derivatives(self):
        permutation = [3, 2, 0, 1]
        geometry = self.geometry(4).detach()[permutation].requires_grad_(True)
        renderer = MitsubaMaskRenderer(
            self.camera(), [4, 3, 1, 2], 128)
        renderer.render(geometry, 19).sum().backward()
        self.assertTrue(torch.all(geometry.grad[:, 6] > 100))

    def test_updated_geometry_keeps_all_radius_derivatives(self):
        geometry = self.geometry(4)
        renderer = MitsubaMaskRenderer(self.camera(), [1, 2, 3, 4], 128)
        for radius in (.08, .09):
            with self.subTest(radius=radius):
                with torch.no_grad():
                    geometry[:, 6] = radius
                geometry.grad = None
                renderer.render(geometry, 23).sum().backward()
                self.assertTrue(torch.all(geometry.grad[:, 6] > 100))


if __name__ == '__main__':
    unittest.main()
