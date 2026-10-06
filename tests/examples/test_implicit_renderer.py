from __future__ import annotations

import math
import unittest

import mitsuba as mi
import numpy as np
import torch

from rebar_inverse import Camera
from rebar_inverse.rendering import MitsubaMaskRenderer


class ImplicitRendererTest(unittest.TestCase):
    def test_end_on_gradients_match_projected_circle(self):
        camera = Camera(
            np.array([[40, 0, 16], [0, 40, 16], [0, 0, 1]]),
            np.eye(4), 32, 32,
        )
        geometry = torch.tensor(
            [[0, 0, 2, 0, 0, 4, .2]], device='cuda',
            requires_grad=True,
        )
        renderer = MitsubaMaskRenderer(camera, [1], 1024)
        renderer.render(geometry, 13).sum().backward()
        expected = 2 * math.pi * (40 / 2) ** 2 * .2
        np.testing.assert_allclose(
            geometry.grad[0, [6, 2]].cpu().numpy(),
            [expected, -expected * .2 / 2], rtol=.15,
        )

    def test_oblique_axis_side_intersection(self):
        camera = Camera(np.eye(3), np.eye(4), 16, 16)
        renderer = MitsubaMaskRenderer(camera, [1], 4)
        start = np.array([.2, -.1, 1.2])
        end = np.array([1.5, .8, 3])
        axis = end - start
        radial = np.cross(axis, [1, 0, 0])
        radial /= np.linalg.norm(radial)
        geometry = torch.tensor(np.r_[start, end, .17][None],
                                dtype=torch.float32, device='cuda')
        with torch.no_grad():
            renderer.render(geometry, 0)
        origin = (start + end) * .5 + 2 * radial
        hit = renderer.scenes[0].ray_intersect(
            mi.Ray3f(mi.Point3f(origin), mi.Vector3f(-radial)))
        self.assertAlmostEqual(float(hit.t[0]), 1.83, places=5)

    def renderer(self):
        camera = Camera(
            np.array([[40, 0, 16], [0, 40, 16], [0, 0, 1]]),
            np.eye(4), 32, 32,
        )
        renderer = MitsubaMaskRenderer(camera, [7], 4)
        geometry = torch.tensor(
            [[0, 0, 2, 0, 0, 4, .5]], device='cuda',
        )
        with torch.no_grad():
            renderer.render(geometry, 0)
        return renderer

    def test_scene_contains_no_triangle_meshes(self):
        renderer = self.renderer()
        self.assertTrue(all(not shape.is_mesh()
                            for shape in renderer.scenes[0].shapes()))

    def test_side_hit_solves_circular_surface_exactly(self):
        renderer = self.renderer()
        angle = math.pi / 48
        direction = np.array([math.cos(angle), math.sin(angle), 0])
        origin = np.array([0, 0, 3]) + 2 * direction
        hit = renderer.scenes[0].ray_intersect(
            mi.Ray3f(mi.Point3f(origin), mi.Vector3f(-direction)))
        self.assertAlmostEqual(float(hit.t[0]), 1.5, places=5)

    def test_endcap_includes_points_between_former_mesh_vertices(self):
        renderer = self.renderer()
        angle = math.pi / 48
        origin = [.4995 * math.cos(angle), .4995 * math.sin(angle), 0]
        hit = renderer.scenes[0].ray_intersect(
            mi.Ray3f(mi.Point3f(origin), mi.Vector3f(0, 0, 1)))
        self.assertAlmostEqual(float(hit.t[0]), 2.0, places=5)

    def test_finite_cylinder_roots(self):
        renderer = self.renderer()
        cases = [
            ('front_cap', [.2, 0, 0], [0, 0, 1], 2),
            ('back_cap', [.2, 0, 5], [0, 0, -1], 1),
            ('inside_exit', [0, 0, 3], [1, 0, 0], .5),
            ('outside_cap', [.6, 0, 0], [0, 0, 1], math.inf),
            ('outside_height', [2, 0, 5], [-1, 0, 0], math.inf),
        ]
        for name, origin, direction, expected in cases:
            with self.subTest(name=name):
                hit = renderer.scenes[0].ray_intersect(
                    mi.Ray3f(mi.Point3f(origin), mi.Vector3f(direction)))
                self.assertAlmostEqual(float(hit.t[0]), expected, places=5)


if __name__ == '__main__':
    unittest.main()
