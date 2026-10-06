"""Real implicit-renderer gradient validation for shared directions."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from rebar_inverse.model import PlanarCylinderModel
from rebar_inverse.rendering import MitsubaMaskRenderer
from rebar_inverse.types import Camera


class SharedDirectionRenderTest(unittest.TestCase):
    """Require both independently parameterized families to receive gradients."""

    def test_renderer_updates_both_shared_angles(self) -> None:
        angles = np.deg2rad([10., 10., 65., 65.])
        axes = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(4)])
        centers = np.array([[-.4, -.45, 3.], [.4, .45, 3.],
                            [-.4, .45, 3.], [.4, -.45, 3.]])
        model = PlanarCylinderModel(
            [1, 2, 3, 4], np.zeros(4, dtype=int),
            np.array([[0., 0., 3.]]), np.eye(3)[None],
            np.stack([centers - axes * .4, centers + axes * .4], axis=1),
            np.full(4, .05), 'cuda', False, np.array([0, 0, 1, 1]))
        camera = Camera(np.array([[80., 0., 32.], [0., 80., 24.],
                                  [0., 0., 1.]]), np.eye(4), 64, 48)
        renderer = MitsubaMaskRenderer(camera, model.instance_ids, 128)
        prediction = renderer.render(model.geometry(), 37)
        y, x = torch.meshgrid(torch.arange(48, device='cuda'),
                              torch.arange(64, device='cuda'), indexing='ij')
        weights = (x * y / (64 * 48)).unsqueeze(-1)
        (prediction * weights).sum().backward()
        self.assertEqual(tuple(model.angles.grad.shape), (2,))
        self.assertTrue(torch.isfinite(model.angles.grad).all())
        self.assertTrue((model.angles.grad.abs() > 1e-4).all())
        torch.optim.SGD(model.parameters(), lr=1e-4).step()
        rods = model.cylinders()
        for first in (0, 2):
            np.testing.assert_allclose(rods[first]['dir'],
                                       rods[first + 1]['dir'], atol=2e-6)
        crossing = np.rad2deg(np.arccos(np.clip(np.dot(
            rods[0]['dir'], rods[2]['dir']), -1, 1)))
        self.assertLess(crossing, 70.)
