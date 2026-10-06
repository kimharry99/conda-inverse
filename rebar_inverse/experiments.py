"""Scene-level data progression, independent of render resolution."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import PurePosixPath
import statistics


class DataUsageProtocol:
    """Build disjoint scene roles from the preserved dataset split lists."""

    @staticmethod
    def _group(frames: list[str]) -> dict[str, list[str]]:
        grouped = defaultdict(list)
        for frame in sorted(set(frames)):
            path = PurePosixPath(frame)
            if path.is_absolute() or len(path.parts) != 2 or '..' in path.parts:
                raise ValueError('Frames must be relative scene/frame paths.')
            grouped[path.parts[0]].append(frame)
        return dict(grouped)

    @staticmethod
    def _rank(value: str, seed: int) -> str:
        return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()

    @staticmethod
    def _stratum(scene: str) -> str:
        parts = scene.split('_')
        return f'{parts[0]}/{parts[2].split("-")[1]}'

    def build(self, training: list[str], holdout: list[str],
              exposed: list[str], per_stratum: int, seed: int) -> dict:
        """Sample train strata and reserve unexposed holdout scenes.

        Args:
            training: Original training scene/frame entries.
            holdout: Original validation scene/frame entries.
            exposed: Frames previously inspected for solver development.
            per_stratum: Positive number of training scenes per condition.
            seed: Fixed hash seed for scene and viewpoint selection.

        Returns:
            PoC, training and test manifests with disjoint scene membership.

        Raises:
            ValueError: Invalid count, overlapping splits or unknown exposure.
        """
        if per_stratum < 1:
            raise ValueError('Need at least one training scene per stratum.')
        train = self._group(training)
        test = self._group(holdout)
        known = self._group(exposed)
        if set(train) & set(test):
            raise ValueError('Original training and holdout scenes overlap.')
        if not set(exposed) <= set(training + holdout):
            raise ValueError('Exposed frames must belong to the source split.')
        strata = defaultdict(list)
        for scene in train:
            if scene not in known:
                strata[self._stratum(scene)].append(scene)
        chosen = []
        for key in sorted(strata):
            ordered = sorted(strata[key], key=lambda s: self._rank(s, seed))
            if len(ordered) < per_stratum:
                raise ValueError(f'Insufficient training scenes in {key}.')
            chosen.extend(ordered[:per_stratum])
        frame_key = lambda p: self._rank(p, seed)
        return {
            'poc': sorted(set(exposed)),
            'training': [min(train[s], key=frame_key) for s in sorted(chosen)],
            'test': [min(test[s], key=frame_key) for s in sorted(test)
                     if s not in known],
            'excluded_test_scenes': sorted(set(test) & set(known)),
            'seed': seed,
        }


class TrainingSelector:
    """Rank a complete training matrix using a predeclared image metric."""

    def select(self, rows: list[dict], frames: list[str],
               candidates: list[str]) -> dict:
        """Choose highest frame-mean IoU, preserving candidate order on ties.

        Only supplied training frames are permitted, exactly once per
        candidate. The caller must persist the result before test execution.
        """
        expected = {(name, frame) for name in candidates for frame in frames}
        scores = {name: [] for name in candidates}
        seen = set()
        for row in rows:
            key = (row['candidate'], row['frame'])
            value = row['selection_iou']
            if row['phase'] != 'training' or key not in expected:
                raise ValueError('Selection only accepts training records.')
            if key in seen or not math.isfinite(value):
                raise ValueError('Duplicate or nonfinite training record.')
            seen.add(key)
            scores[key[0]].append(value)
        if seen != expected or not expected:
            raise ValueError('Training matrix must be complete before test.')
        means = {name: statistics.mean(values)
                 for name, values in scores.items()}
        return {'selected': max(candidates, key=lambda name: means[name]),
                'training_scores': means,
                'criterion': 'mean per-frame IoU, target coverage >=4 pixels'}
