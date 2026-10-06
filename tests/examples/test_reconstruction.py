from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rebar_inverse import Camera
from rebar_inverse import Observation
from rebar_inverse import ReconstructionConfig


class ReconstructionTest(unittest.TestCase):
    def test_v2_multiview_rejects_unverified_poses(self):
        from rebar_inverse.dataset import V2Dataset
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            manifest = root / 'frames.txt'
            manifest.write_text('scene/frame_0000\nscene/frame_0001\n')
            with self.assertRaisesRegex(ValueError, 'explicit poses'):
                V2Dataset(root).load_manifest(manifest)

    def test_background_view_has_serializable_metrics(self):
        from rebar_inverse import RebarReconstructor
        mask = np.zeros((32, 32), dtype=np.int32)
        mask[12:16, 5:27] = 1
        k = np.array([[35, 0, 16], [0, 35, 16], [0, 0, 1]])
        pose = np.eye(4)
        pose[0, 3] = 5
        views = [Observation(mask, Camera(k, np.eye(4), 32, 32)),
                 Observation(np.zeros_like(mask), Camera(k, pose, 32, 32))]
        result = RebarReconstructor(ReconstructionConfig(
            max_sizes=(32,), iterations=(1,), spp=8)).reconstruct(views)
        with tempfile.TemporaryDirectory() as output:
            result.save(output)
        self.assertIsNone(result.metrics['final']['views'][1]['mean_iou'])

    def test_optimizer_improves_masks(self):
        try:
            from rebar_inverse import RebarReconstructor
        except ImportError as error:
            self.fail(f'Reconstruction entry point missing: {error}')
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[15:20, 8:56] = 5
        mask[42:47, 8:56] = 9
        camera = Camera(
            np.array([[75, 0, 32], [0, 75, 32], [0, 0, 1]]),
            np.eye(4), 64, 64,
        )
        config = ReconstructionConfig(
            max_sizes=(32, 64), iterations=(8, 12), spp=64,
            learning_rate=.003,
        )
        result = RebarReconstructor(config).reconstruct(
            [Observation(mask, camera, 'synthetic')],
            mesh_ids={5: 0, 9: 0},
            provenance={'mesh_labels': {'source': 'gt'}},
        )
        self.assertTrue(result.provenance['gt_used_for_initialization'])
        self.assertFalse(
            result.provenance['gt_geometry_used_as_initial_parameters'])
        self.assertLess(result.metrics['final']['loss'],
                        result.metrics['initial']['loss'])
        self.assertGreater(result.metrics['final']['mean_iou'], .75)
        self.assertEqual({row['max_size'] for row in result.history}, {32, 64})
        self.assertEqual(len(result.cylinders), 2)
        with tempfile.TemporaryDirectory() as output:
            result.save(Path(output))
            saved = json.loads((Path(output) / 'result.json').read_text())
            self.assertEqual(saved['gauge']['units'], 'arbitrary')
            self.assertIn('source_sha256', saved['provenance'])

    def test_v2_loader_uses_camera_frame_without_gt(self):
        try:
            from rebar_inverse.dataset import V2Dataset
        except ImportError as error:
            self.fail(f'v2 dataset adapter missing: {error}')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / 'scene' / 'frame_0000'
            frame.mkdir(parents=True)
            np.save(frame / 'instance_id.npy', np.ones((8, 12), np.int32))
            (frame / 'meta.json').write_text(json.dumps({
                'intrinsics': [[2, 0, .5], [0, 3, .5], [0, 0, 1]],
                'camera': {'width': 12, 'height': 8,
                           'location': [50, 20, 10], 'look_at': [0, 0, 0]},
            }))
            observation = V2Dataset(root).load_frame('scene/frame_0000')
            np.testing.assert_allclose(observation.camera.intrinsic,
                                       [[24, 0, 6], [0, 24, 4], [0, 0, 1]])
            np.testing.assert_array_equal(observation.camera.camera_to_world,
                                          np.eye(4))


if __name__ == '__main__':
    unittest.main()
