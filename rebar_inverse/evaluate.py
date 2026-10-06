"""Evaluate a saved reconstruction against one v2_2layer GT camera frame."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from rebar_inverse.dataset import V2Dataset
from rebar_inverse.evaluation import MeshScaleEvaluator
from rebar_inverse.types import Camera
from rebar_inverse.types import Observation


class V2Evaluation:
    """Read saved geometry, apply known camera coordinates, and fit scales."""

    def evaluate(self, result_path: str | Path,
                 dataset_root: str | Path = 'data/rebar/v2_2layer',
                 frame: str | None = None) -> dict:
        """Evaluate mask-visible IDs in one frame using predicted mesh IDs.

        Known pose conversion is a coordinate change, not an alignment fitted
        to GT. Multiple input frames require selecting one evaluation frame.
        The selected frame must have a saved normalized camera pose.
        """
        result_path = Path(result_path).resolve()
        source = json.loads(result_path.read_text(encoding='utf-8'))
        provenance = source['provenance']
        frames = provenance['dataset']['frames']
        if frame is None:
            if len(frames) != 1:
                raise ValueError('Select --frame for a multiview result.')
            frame = frames[0]
        if frame not in frames:
            raise ValueError('Evaluation frame must be an input observation.')
        dataset = V2Dataset(dataset_root)
        directory = dataset.frame_path(frame)
        cameras = [camera for camera in provenance['normalized_cameras']
                   if camera['name'] == frame]
        if len(cameras) != 1:
            raise ValueError('A unique normalized camera is required.')
        saved_camera = cameras[0]
        camera = Camera(np.array(saved_camera['intrinsic']),
                        np.array(saved_camera['camera_to_world']),
                        saved_camera['width'], saved_camera['height'])
        mask_path = directory / 'instance_id.npy'
        gt_path = directory / 'gt_cylinders.json'
        checksums = provenance['dataset'].get('input_sha256', {})
        mask_hash = self._sha256(mask_path)
        expected_hash = checksums.get(f'{frame}/instance_id.npy')
        if expected_hash is not None and mask_hash != expected_hash:
            raise ValueError('Mask checksum differs from reconstruction input.')
        mask = np.load(mask_path, allow_pickle=False)
        Observation(mask, camera, frame)
        ids = np.unique(mask[mask > 0]).tolist()
        truth = json.loads(gt_path.read_text(encoding='utf-8'))
        rotation = camera.camera_to_world[:3, :3]
        origin = camera.camera_to_world[:3, 3]
        predicted = copy.deepcopy(source['cylinders'])
        for rod in predicted:
            rod['mid'] = (rotation.T @ (np.array(rod['mid']) - origin)).tolist()
            rod['dir'] = (rotation.T @ np.array(rod['dir'])).tolist()
        report = MeshScaleEvaluator().evaluate(predicted, truth, ids)
        report['method']['origin'] = 'evaluation camera center'
        report['method']['coordinate_conversion'] = (
            'R_cw.T @ (p_world - camera_center); known pose, never GT-fitted')
        report['provenance'] = {
            'result_path': str(result_path),
            'result_sha256': self._sha256(result_path),
            'dataset_root': str(dataset.root), 'frame': frame,
            'gt_path': str(gt_path), 'gt_sha256': self._sha256(gt_path),
            'mask_sha256': mask_hash,
            'evaluation_source_sha256': {
                name: self._sha256(Path(__file__).with_name(name))
                for name in ('evaluation.py', 'evaluate.py')},
            'reconstruction_config': source.get('config', {}),
            'reconstruction_provenance': provenance,
            'normalized_evaluation_camera': saved_camera,
            'gt_coordinates': 'camera +X right, +Y down, +Z front; meters',
        }
        scales = {mesh['mesh_id']: mesh['scale'] for mesh in report['meshes']}
        report['aligned_planes'] = []
        for plane in source.get('planes', []):
            mesh_id = plane['mesh_id']
            if mesh_id in scales:
                report['aligned_planes'].append({
                    'mesh_id': mesh_id,
                    'origin': (scales[mesh_id] * rotation.T
                               @ (np.array(plane['origin']) - origin)).tolist(),
                    'normal': (rotation.T @ np.array(plane['normal'])).tolist(),
                })
        report['warnings'] = source.get('warnings', []) + [
            'GT-assisted posthoc scales are not recovered metric depth.',
            'Full GT extent includes hidden/offscreen endpoints.',
            'Predicted mesh memberships are fixed; GT does not regroup rods.',
            'Independent scales remove inter-mesh distance/order constraints.',
        ]
        report['gt_plane_diagnostics'] = self._gt_planes(
            truth, predicted, ids)
        counts = report['gt_plane_diagnostics'].get(
            'predicted_mesh_plane_counts', {})
        for mesh_id, planes in counts.items():
            report['warnings'].append(
                f'Predicted mesh {mesh_id} contains rods from '
                f'{len(planes)} distinct GT center-axis planes. '
                'These are geometric planes, not semantic layer labels.')
        return report

    @staticmethod
    def _gt_planes(truth: list[dict], predicted: list[dict],
                   visible_ids: list[int]) -> dict:
        directions = np.array([rod['dir'] for rod in truth], dtype=float)
        _, singular, axes = np.linalg.svd(directions, full_matrices=False)
        if (len(singular) < 3 or singular[1] < singular[0] * 1e-6
                or singular[2] > singular[0] * 1e-6):
            return {'available': False,
                    'reason': 'GT axes do not define a common plane normal.'}
        normal = axes[-1]
        if normal[np.argmax(np.abs(normal))] < 0:
            normal = -normal
        offsets = np.array([rod['mid'] for rod in truth]) @ normal
        tolerance = 1e-5
        groups = []
        for index in np.argsort(offsets):
            if (not groups or offsets[index]
                    - offsets[groups[-1][0]] > tolerance):
                groups.append([])
            groups[-1].append(int(index))
        planes, membership = [], {}
        for plane_id, indices in enumerate(groups):
            ids = [truth[i]['instance_id'] for i in indices]
            membership.update({instance_id: plane_id for instance_id in ids})
            planes.append({'plane_id': plane_id,
                           'offset_m': float(np.mean(offsets[indices])),
                           'instance_ids': sorted(ids)})
        counts = {}
        for rod in predicted:
            if rod['instance_id'] in visible_ids:
                mesh = counts.setdefault(str(rod['mesh_id']), {})
                plane_id = str(membership[rod['instance_id']])
                mesh[plane_id] = mesh.get(plane_id, 0) + 1
        return {
            'available': True, 'normal_camera': normal.tolist(),
            'offset_tolerance_m': tolerance, 'center_planes': planes,
            'predicted_mesh_plane_counts': counts,
            'note': 'Geometric axis planes only; no semantic regrouping.',
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def save(self, report: dict, output: str | Path) -> Path:
        """Save alignment JSON, per-instance CSV and a Markdown report."""
        output = Path(output).resolve()
        protected_roots = [
            Path('data/rebar').resolve(),
            Path(report['provenance']['dataset_root']).resolve(),
        ]
        for protected in protected_roots:
            if output == protected or protected in output.parents:
                raise ValueError('Evaluation outputs must be outside datasets.')
        output.mkdir(parents=True, exist_ok=True)
        path = output / 'alignment.json'
        if path.resolve() == Path(report['provenance']['result_path']):
            raise ValueError('Cannot overwrite the source reconstruction.')
        path.write_text(json.dumps(report, indent=2, allow_nan=False),
                        encoding='utf-8')
        with (output / 'per_instance.csv').open('w', newline='',
                                                encoding='utf-8') as stream:
            writer = csv.DictWriter(stream,
                                    fieldnames=list(report['per_instance'][0]))
            writer.writeheader()
            writer.writerows(report['per_instance'])
        (output / 'report.md').write_text(self._markdown(report),
                                         encoding='utf-8')
        return path

    @staticmethod
    def _markdown(report: dict) -> str:
        lines = [
            '# 철근망별 스케일 정렬 및 GT 오차', '',
            f"대상: `{report['provenance']['frame']}`", '',
            '각 예측 mesh_id에 x′ = s x인 양의 균일 스케일 하나를 적용합니다.',
            's = Σ(p·g) / Σ(p·p); visible 철근 중심에 동일 가중치를 줍니다.',
            '길이와 반지름에도 같은 s를 곱합니다. 이동·회전은 추정하지 않습니다.',
            '평가 단계에서 GT로 스케일을 정렬하며 최적화 원본은 보존합니다.', '',
            '주 위치 지표는 예측 중심에서 GT 무한 축까지의 수직 거리입니다.',
            'e⊥ = ||δ − (δ·d_GT)d_GT||; δ = s p − g.',
            '철근별 거리를 제곱 평균한 뒤 제곱근으로 RMSE를 계산합니다.', '',
            '| 망 | 철근 수 | 스케일 (m/임의단위) | 수직 축 RMSE (mm) |'
            ' Line-to-line RMSE (mm) | 지름 MAE (mm) | 축 오차 (°) |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
        ]
        groups = [(str(mesh['mesh_id']), f"{mesh['scale']:.8f}",
                   mesh['metrics']) for mesh in report['meshes']]
        groups.append(('전체', '—', report['overall']))
        for label, scale, metrics in groups:
            lines.append(
                f"| {label} | {metrics['count']} | {scale} | "
                f"{metrics['perpendicular_axis_rmse_mm']:.3f} | "
                f"{metrics['line_to_line_rmse_mm']:.3f} | "
                f"{metrics['diameter_mae_mm']:.3f} | "
                f"{metrics['axis_mean_deg']:.3f} |")
        lines.extend([
            '', 'Line-to-line은 두 무한 축의 최소 거리입니다. 교차하면 0이므로',
            '수직 거리와 축 각도도 함께 봐야 합니다. 평행 판정은',
            '||d_pred × d_GT|| ≤ 1e-10이며 이때 수직 거리로 계산합니다.',
            '비평행 축의 수직 거리는 예측 중심의 위치에 의존하는 비대칭 지표입니다.',
            '스케일 fitting은 기존 중심 최소제곱을 유지하므로 길이 잘림에 따른',
            '중심 오차가 여전히 s에 영향을 줄 수 있습니다.', '',
            '보조 진단 (전체 길이와 중심의 영향):', '',
            '| 망 | 축 방향 중심 RMSE (mm) | 중심 RMSE (mm) |'
            ' 끝점 RMSE (mm) | 길이 MAE (mm) |',
            '| --- | ---: | ---: | ---: | ---: |',
        ])
        for label, _, metrics in groups:
            lines.append(
                f"| {label} | {metrics['along_axis_center_rmse_mm']:.3f} | "
                f"{metrics['center_rmse_mm']:.3f} | "
                f"{metrics['endpoint_rmse_mm']:.3f} | "
                f"{metrics['length_mae_mm']:.3f} |")
        lines.extend([
            '', '축 오차는 d와 −d를 동등하게 처리한 평균 각도입니다.',
            '끝점 RMSE는 끝점 순서를 바꿔 비교한 최소 오차이며, 전체 길이를',
            '평가합니다. 전체 지표는 모든 철근을 같은 가중치로 집계합니다.',
            '', '유의사항:', '',
        ])
        lines.extend(f'- {warning}' for warning in report['warnings'])
        lines.extend(['', '상세 값: [JSON](alignment.json), '
                      '[철근별 CSV](per_instance.csv).', ''])
        return '\n'.join(lines)

    @classmethod
    def main(cls) -> None:
        """Run reproducible evaluation with python -m rebar_inverse.evaluate."""
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--result', required=True, type=Path)
        parser.add_argument('--output', required=True, type=Path)
        parser.add_argument('--dataset-root', type=Path,
                            default=Path('data/rebar/v2_2layer'))
        parser.add_argument('--frame')
        args = parser.parse_args()
        evaluator = cls()
        report = evaluator.evaluate(args.result, args.dataset_root, args.frame)
        path = evaluator.save(report, args.output)
        print(json.dumps({'alignment': str(path), 'overall': report['overall'],
                          'scales': {mesh['mesh_id']: mesh['scale']
                                     for mesh in report['meshes']}}, indent=2))


if __name__ == '__main__':
    V2Evaluation.main()
