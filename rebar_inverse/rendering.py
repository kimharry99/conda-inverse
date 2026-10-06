"""Exact implicit-cylinder ray intersections and visibility derivatives."""
from __future__ import annotations

import math

import drjit as dr
import mitsuba as mi
import numpy as np
import torch
from torch.nn import functional as functional

from rebar_inverse.types import Camera


class _SceneBatchFunction(torch.autograd.Function):
    """Replay and finish each RGB adjoint before changing shared scene state."""

    @staticmethod
    def forward(ctx, transforms, renderer, seed):
        ctx.save_for_backward(transforms.detach())
        ctx.renderer = renderer
        ctx.seed = seed
        return torch.cat([
            renderer._render_pass(transforms.detach(), group, seed)
            for group in range(renderer.pass_count)
        ], dim=2)

    @staticmethod
    def backward(ctx, adjoint):
        saved, = ctx.saved_tensors
        gradient = torch.zeros_like(saved)
        with torch.enable_grad():
            for group in range(ctx.renderer.pass_count):
                transforms = saved.detach().requires_grad_(True)
                image = ctx.renderer._render_pass(transforms, group, ctx.seed)
                piece = adjoint[:, :, group * 3:(group + 1) * 3].contiguous()
                derivative, = torch.autograd.grad(image, transforms, piece)
                gradient += derivative.detach()
        return gradient, None, None


class MitsubaMaskRenderer:
    """Render all implicit cylinders through one scene and one mask batch.

    Each rod is a quadratic cylinder side with two plane/disk endcaps.
    RGB packing uses multiple passes over the same complete geometry.
    The batch backward replays and finishes each pass before changing
    scene colors, preserving every occluder's visibility derivative.
    Finish backward before reusing this mutable renderer.
    """

    def __init__(self, camera: Camera, instance_ids: list[int],
                 spp: int = 16) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required.')
        if spp < 1 or not instance_ids:
            raise ValueError('Need instance IDs and positive spp.')
        mi.set_variant('cuda_ad_rgb')
        self.camera = camera
        self.instance_ids = instance_ids
        self.spp = spp
        self.vertical_scale = camera.intrinsic[0, 0] / camera.intrinsic[1, 1]
        self.render_height = math.ceil(camera.height * self.vertical_scale)
        if not 0.125 <= self.vertical_scale <= 8:
            raise ValueError('Rectify extreme pixel aspect ratios first.')
        self.pass_count = math.ceil(len(instance_ids) / 3)
        self.shape_names = [
            f'rod_{instance_id}_{part}' for instance_id in instance_ids
            for part in ('side', 'start', 'end')]
        scene = {
            'type': 'scene', 'sensor': self._sensor(),
            'integrator': {'type': 'prb_projective', 'max_depth': 1,
                           'sppi': 0, 'sppc': 0},
        }
        for instance_id in instance_ids:
            for part in ('side', 'start', 'end'):
                scene[f'rod_{instance_id}_{part}'] = {
                    'type': 'cylinder' if part == 'side' else 'disk',
                    'bsdf': {'type': 'diffuse', 'reflectance': 0.0},
                    'flip_normals': part == 'start',
                    'silhouette_sampling_weight': (
                        1.0 if part == 'side' else 0.0),
                    'emitter': {'type': 'area', 'radiance': {
                        'type': 'rgb', 'value': [0., 0., 0.]}},
                }
        self.scenes = [mi.load_dict(scene)]
        parameters = mi.traverse(self.scenes[0])
        self.color_keys = [f'{name}.emitter.radiance.value'
                           for name in self.shape_names]
        parameters.keep([f'{name}.to_world' for name in self.shape_names]
                        + self.color_keys)
        self.parameters = [parameters]

    @staticmethod
    def _transforms(geometry: torch.Tensor) -> torch.Tensor:
        """Map unit implicit surfaces to rods; inverse transforms stay in AD.

        Constructing both matrices in PyTorch keeps the Dr.Jit bridge free
        of per-shape matrix inversion graphs without detaching geometry.
        """
        delta = geometry[:, 3:6] - geometry[:, :3]
        length = torch.linalg.vector_norm(delta, dim=1)
        direction = delta / length[:, None]
        axis = direction.detach().abs().argmin(dim=1)
        reference = functional.one_hot(axis, 3).to(geometry.dtype)
        u = functional.normalize(torch.linalg.cross(reference, direction),
                                 dim=1)
        v = torch.linalg.cross(direction, u)
        basis = torch.stack([u, v, direction], dim=2)
        transforms = []
        for part in range(3):
            origin = geometry[:, 3:6] if part == 2 else geometry[:, :3]
            axial = length if part == 0 else torch.ones_like(length)
            scale = torch.stack([geometry[:, 6], geometry[:, 6], axial], 1)
            matrix = geometry.new_zeros((len(geometry), 4, 4))
            matrix[:, :3, :3] = basis * scale[:, None]
            matrix[:, :3, 3] = origin
            matrix[:, 3, 3] = 1
            inverse_transpose = geometry.new_zeros(matrix.shape)
            inverse_linear = basis / scale[:, None]
            inverse_transpose[:, :3, :3] = inverse_linear
            inverse_transpose[:, 3, :3] = -torch.einsum(
                'ni,nij->nj', origin, inverse_linear)
            inverse_transpose[:, 3, 3] = 1
            transforms.append(torch.stack([matrix, inverse_transpose], 1))
        return torch.stack(transforms, 1).reshape(-1, 2, 4, 4)

    def _sensor(self) -> dict:
        k = self.camera.intrinsic
        width = self.camera.width
        height = self.render_height
        pose = self.camera.camera_to_world @ np.diag([-1, -1, 1, 1])
        return {
            'type': 'perspective',
            'to_world': mi.ScalarAffineTransform4f(pose),
            'fov_axis': 'x',
            'fov': math.degrees(2 * math.atan(width / (2 * k[0, 0]))),
            'principal_point_offset_x': 0.5 - k[0, 2] / width,
            'principal_point_offset_y': (
                0.5 - k[1, 2] * self.vertical_scale / height),
            'near_clip': 1e-5, 'far_clip': 1e5,
            'film': {
                'type': 'hdrfilm', 'width': width, 'height': height,
                'pixel_format': 'rgb', 'component_format': 'float32',
                'sample_border': True, 'rfilter': {'type': 'box'},
            },
            'sampler': {'type': 'independent', 'sample_count': self.spp},
        }

    @dr.wrap(source='torch', target='drjit')
    def _render_pass(self, transforms, group: int, seed: int):
        """Render one RGB pass with every cylinder retained as an occluder."""
        offsets = dr.arange(mi.UInt, len(self.shape_names)) * 32
        matrix = mi.Matrix4f([
            [dr.gather(mi.Float, transforms.array, offsets + r * 4 + c)
             for c in range(4)] for r in range(4)
        ])
        inverse_transpose = mi.Matrix4f([
            [dr.gather(mi.Float, transforms.array, offsets + 16 + r * 4 + c)
             for c in range(4)] for r in range(4)
        ])
        parameters = self.parameters[0]
        for index, name in enumerate(self.shape_names):
            parameters[f'{name}.to_world'] = mi.AffineTransform4f(
                dr.gather(mi.Matrix4f, matrix, index),
                dr.gather(mi.Matrix4f, inverse_transpose, index))
            color = [float(index // 3 == group * 3 + c) for c in range(3)]
            parameters[self.color_keys[index]] = mi.Color3f(color)
        parameters.update()
        return mi.render(self.scenes[0], parameters, seed=seed + group,
                         spp=self.spp, spp_grad=self.spp)

    def _render_batch(self, transforms, seed: int):
        return (_SceneBatchFunction.apply(transforms, self, seed),)

    def render(self, geometry: torch.Tensor, seed: int) -> torch.Tensor:
        """Render H x W x N coverage with gradients to N x 7 geometry."""
        if geometry.shape != (len(self.instance_ids), 7):
            raise ValueError('Expected one [p0, p1, radius] row per instance.')
        if geometry.device.type != 'cuda':
            raise ValueError('Renderer geometry must be on CUDA.')
        transforms = self._transforms(geometry)
        images = self._render_batch(transforms, seed)
        image = torch.cat(images, dim=2)[:, :, :len(self.instance_ids)]
        if (self.render_height == self.camera.height
                and abs(self.vertical_scale - 1) < 1e-7):
            return image
        y = (torch.arange(self.camera.height, device=geometry.device) + .5)
        y = y * self.vertical_scale / self.render_height * 2 - 1
        x = (torch.arange(self.camera.width, device=geometry.device) + .5)
        x = x / self.camera.width * 2 - 1
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([xx, yy], dim=-1)[None]
        resampled = functional.grid_sample(
            image.permute(2, 0, 1)[None], grid, align_corners=False,
            padding_mode='border',
        )
        return resampled[0].permute(1, 2, 0)
