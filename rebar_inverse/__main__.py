"""Manifest-based v2_2layer reconstruction command."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rebar_inverse import RebarReconstructor
from rebar_inverse import ReconstructionConfig
from rebar_inverse.dataset import V2Dataset


class ReconstructionCommand:
    """Run a fixed scene/frame manifest and save reconstruction metadata."""

    @staticmethod
    def progress(record: dict) -> None:
        print(json.dumps(record), flush=True)

    @classmethod
    def main(cls) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--dataset-root',
                            default='data/rebar/v2_2layer')
        parser.add_argument('--manifest', required=True, type=Path)
        parser.add_argument('--output', required=True, type=Path)
        parser.add_argument('--poses', type=Path,
                            help='JSON scene/frame -> camera_to_world')
        membership = parser.add_mutually_exclusive_group()
        membership.add_argument('--mesh-ids', type=Path,
                                help='JSON instance ID -> common-plane label')
        membership.add_argument('--mesh-label-source', choices=['gt', 'mask'],
                                default='gt', help='Default: GT membership')
        parser.add_argument('--max-sizes', nargs='+', type=int,
                            default=[128, 256, 512])
        parser.add_argument('--iterations', nargs='+', type=int,
                            default=[30, 30, 40])
        parser.add_argument('--planes', type=int, default=2,
                            help='Plane count for mask-inferred membership')
        parser.add_argument('--spp', type=int, default=64)
        parser.add_argument('--seed', type=int, default=7)
        parser.add_argument('--learning-rate', type=float, default=.001)
        parser.add_argument('--independent-directions', action='store_true',
                            help='Disable direction sharing within each mesh')
        args = parser.parse_args()
        dataset = V2Dataset(args.dataset_root)
        output = args.output.resolve()
        if output == dataset.root or dataset.root in output.parents:
            parser.error('Output must be outside the source dataset.')
        poses = None
        if args.poses:
            poses = {key: np.asarray(value) for key, value in
                     json.loads(args.poses.read_text()).items()}
        mesh_ids = None
        if args.mesh_ids:
            mesh_ids = {int(key): int(value) for key, value in
                        json.loads(args.mesh_ids.read_text()).items()}
        config = ReconstructionConfig(
            max_sizes=tuple(args.max_sizes),
            iterations=tuple(args.iterations),
            learning_rate=args.learning_rate, spp=args.spp,
            seed=args.seed, num_planes=args.planes,
            share_directions=not args.independent_directions,
        )
        views = dataset.load_manifest(args.manifest, poses)
        provenance = dataset.provenance(args.manifest)
        if mesh_ids is not None:
            provenance['mesh_labels'] = {
                'source': 'provided', 'instance_to_mesh': mesh_ids}
        elif args.mesh_label_source == 'gt':
            mesh_ids, provenance['mesh_labels'] = dataset.gt_mesh_labels(views)
        else:
            provenance['mesh_labels'] = {'source': 'mask'}
        result = RebarReconstructor(config).reconstruct(
            views, mesh_ids, cls.progress, provenance)
        result.save(output)
        if mesh_ids is not None:
            (output / 'mesh_labels.json').write_text(
                json.dumps(mesh_ids, indent=2), encoding='utf-8')
        (output / 'sample_manifest.txt').write_bytes(args.manifest.read_bytes())
        print(json.dumps(result.metrics, indent=2), flush=True)


if __name__ == '__main__':
    ReconstructionCommand.main()
