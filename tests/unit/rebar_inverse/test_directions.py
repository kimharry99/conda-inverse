"""Behavioral checks for shared, non-orthogonal direction families."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from rebar_inverse.model import PlanarCylinderModel


class SharedDirectionTest(unittest.TestCase):
    """Exercise parameter updates without a renderer or GT geometry."""

    @staticmethod
    def model(shared: bool = True,
              degrees: list[float] | None = None) -> PlanarCylinderModel:
        angles = np.deg2rad(degrees if degrees is not None else
                           [0, 4, 60, 62, 10, 14, 70, 74])
        direction = np.column_stack([
            np.cos(angles), np.sin(angles), np.zeros(8)])
        center = np.column_stack([np.arange(8), np.zeros(8), np.ones(8)])
        return PlanarCylinderModel(
            list(range(1, 9)), np.repeat([0, 1], 4),
            np.array([[0., 0., 1.], [0., 0., 1.]]),
            np.tile(np.eye(3), (2, 1, 1)),
            np.stack([center - direction, center + direction], axis=1),
            np.full(8, .01), 'cpu', False,
            direction_families=(np.tile([0, 0, 1, 1], 2)
                                if shared else None))

    @staticmethod
    def axes(model: PlanarCylinderModel) -> np.ndarray:
        geometry = model.geometry().detach().numpy()
        delta = geometry[:, 3:6] - geometry[:, :3]
        return delta / np.linalg.norm(delta, axis=1)[:, None]

    def test_group_parallelism_survives_updates(self) -> None:
        model = self.model()
        self.assertEqual(model.angles.numel(), 4)
        with torch.no_grad():
            model.angles.add_(torch.tensor([.1, -.2, .3, -.4]))
            model.plane_tilts.add_(torch.tensor([[.2, .1], [-.1, .3]]))
            model.log_lengths.add_(torch.arange(8) * .03)
        axes = self.axes(model)
        for first in (0, 2, 4, 6):
            np.testing.assert_allclose(axes[first], axes[first + 1],
                                       atol=2e-6)

    def test_family_angle_is_not_forced_to_ninety_degrees(self) -> None:
        model = self.model()
        axes = self.axes(model)
        angle = np.rad2deg(np.arccos(np.dot(axes[0], axes[2])))
        self.assertAlmostEqual(angle, 59., places=4)
        before = axes.copy()
        with torch.no_grad():
            model.angles[0].add_(.1)
        after = self.axes(model)
        self.assertGreater(np.linalg.norm(after[0] - before[0]), .09)
        np.testing.assert_allclose(after[2:], before[2:], atol=1e-6)

    def test_each_member_contributes_to_shared_gradient(self) -> None:
        model = self.model()
        geometry = model.geometry()
        loss = geometry[0, 4] + 2 * geometry[1, 4]
        loss.backward()
        self.assertGreater(float(model.angles.grad[0]), 2.9)
        np.testing.assert_array_equal(model.angles.grad[1:].numpy(), 0)

    def test_independent_mode_preserves_individual_angles(self) -> None:
        model = self.model(False)
        self.assertEqual(model.angles.numel(), 8)
        axes = self.axes(model)
        np.testing.assert_allclose(
            np.rad2deg(np.arctan2(axes[:, 1], axes[:, 0])),
            [0, 4, 60, 62, 10, 14, 70, 74], atol=2e-5)

    def test_axial_median_handles_pi_wrap(self) -> None:
        model = self.model(degrees=[179, -179, 59, 61, -179, 179, -119, 61])
        axes = self.axes(model)
        for first, expected in ((0, 0.), (2, 60.), (4, 0.), (6, 61.)):
            direction = np.array([np.cos(np.deg2rad(expected)),
                                  np.sin(np.deg2rad(expected)), 0.])
            self.assertAlmostEqual(abs(float(axes[first] @ direction)),
                                   1., places=6)
            np.testing.assert_allclose(axes[first], axes[first + 1],
                                       atol=2e-6)

    def test_axis_sign_does_not_split_families(self) -> None:
        from rebar_inverse.directions import DirectionFamilies
        for degrees, expected_count in (
                ([0, 180, 60, 240], 2),
                ([0, 180, 0, 180], 1), ([20], 1)):
            with self.subTest(degrees=degrees):
                labels = DirectionFamilies.infer(
                    np.deg2rad(degrees), np.zeros(len(degrees), dtype=int))
                self.assertEqual(len(np.unique(labels)), expected_count)
                if len(degrees) > 1:
                    self.assertEqual(labels[0], labels[1])
                if expected_count == 2:
                    self.assertEqual(labels[2], labels[3])
                    self.assertNotEqual(labels[0], labels[2])


class DirectionInitializationTest(unittest.TestCase):
    """Check the public initialization path using only synthetic masks."""

    def test_configuration_selects_shared_direction_parameters(self) -> None:
        from rebar_inverse.initialization import MaskInitializer
        from rebar_inverse.types import Camera
        from rebar_inverse.types import Observation
        from rebar_inverse.types import ReconstructionConfig
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[12:15, 5:59] = 1
        mask[45:48, 5:59] = 2
        mask[5:59, 21:24] = 3
        mask[5:59, 37:40] = 4
        camera = Camera(np.array([[70., 0., 32.], [0., 70., 32.],
                                  [0., 0., 1.]]), np.eye(4), 64, 64)
        for shared, count in ((True, 2), (False, 4)):
            with self.subTest(shared=shared):
                state = MaskInitializer(ReconstructionConfig(
                    device='cpu', share_directions=shared)).initialize(
                        [Observation(mask, camera)], {i: 0 for i in range(1, 5)})
                self.assertEqual(state.model.angles.numel(), count)
                self.assertFalse(state.gauge['direction_constraint'][
                    'orthogonality_constraint'])
                rods = state.model.cylinders()
                if shared:
                    self.assertEqual(rods[0]['direction_family'],
                                     rods[1]['direction_family'])
                    self.assertEqual(rods[2]['direction_family'],
                                     rods[3]['direction_family'])
                    self.assertNotEqual(rods[0]['direction_family'],
                                        rods[2]['direction_family'])
                else:
                    self.assertNotIn('direction_family', rods[0])

    def test_shared_direction_constraint_is_enabled_by_default(self) -> None:
        from rebar_inverse.types import ReconstructionConfig
        self.assertTrue(ReconstructionConfig().share_directions)
