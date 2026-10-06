"""Tests for scene-held-out data progression and training-only selection."""
from __future__ import annotations

import unittest

from rebar_inverse.experiments import DataUsageProtocol
from rebar_inverse.experiments import TrainingSelector


class DataUsageProtocolTest(unittest.TestCase):
    """Keep all views of exposed scenes away from final evaluation."""

    def setUp(self) -> None:
        self.train = [f'Env-1_D10_L1-normal_s{i}/frame_000{v}'
                      for i in range(4) for v in range(2)]
        self.holdout = [f'Env-2_D10_L1-normal_s{i}/frame_000{v}'
                        for i in range(3) for v in range(2)]
        self.exposed = [self.holdout[0]]

    def test_exposed_scene_is_excluded(self) -> None:
        split = DataUsageProtocol().build(
            self.train, self.holdout, self.exposed, 2, 928)
        self.assertEqual(len(split['training']), 2)
        self.assertEqual(len(split['test']), 2)
        test_scenes = {p.split('/')[0] for p in split['test']}
        self.assertNotIn(self.exposed[0].split('/')[0], test_scenes)
        self.assertEqual(split['poc'], self.exposed)

    def test_input_order_does_not_change_selection(self) -> None:
        protocol = DataUsageProtocol()
        actual = protocol.build(self.train, self.holdout, self.exposed, 2, 928)
        reordered = protocol.build(
            self.train[::-1], self.holdout[::-1], self.exposed, 2, 928)
        self.assertEqual(actual, reordered)
        self.assertEqual(len(actual['training']), 2)

    def test_sample_count_boundary(self) -> None:
        with self.assertRaises(ValueError):
            DataUsageProtocol().build(
                self.train, self.holdout, self.exposed, 0, 928)
        for count in (1, 2):
            with self.subTest(count=count):
                split = DataUsageProtocol().build(
                    self.train, self.holdout, self.exposed, count, 928)
                self.assertEqual(len(split['training']), count)

    def test_overlapping_source_scenes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DataUsageProtocol().build(
                self.train, self.holdout + [self.train[1]],
                self.exposed, 2, 928)


class TrainingSelectorTest(unittest.TestCase):
    """Choose one common configuration from a complete training matrix."""

    def setUp(self) -> None:
        self.rows = [
            {'phase': 'training', 'candidate': name, 'frame': frame,
             'selection_iou': value}
            for name, value in [('baseline', .6), ('small_step', .7)]
            for frame in ('scene1/frame_0', 'scene2/frame_0')]
        self.frames = ['scene1/frame_0', 'scene2/frame_0']
        self.candidates = ['baseline', 'small_step']

    def test_highest_training_mean_is_selected(self) -> None:
        result = TrainingSelector().select(
            self.rows, self.frames, self.candidates)
        self.assertEqual(result['selected'], 'small_step')

    def test_test_record_is_rejected(self) -> None:
        self.rows[0]['phase'] = 'test'
        with self.assertRaises(ValueError):
            TrainingSelector().select(
                self.rows, self.frames, self.candidates)

    def test_incomplete_matrix_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrainingSelector().select(
                self.rows[:-1], self.frames, self.candidates)


if __name__ == '__main__':
    unittest.main()
