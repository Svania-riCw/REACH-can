"""Paired-first execution selects independent policies and CPU detector jobs."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_paired_first_lane import lane_plan


class PairedLaneTests(unittest.TestCase):
    def test_policies_have_disjoint_ports_and_same_experiment_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            plans = [lane_plan(mode, 0, Path(folder)) for mode in ('baseline', 'conditional')]
        jobs = [p['training_jobs'][0] for p in plans]
        self.assertTrue(all(len(p['training_jobs']) == 1 for p in plans))
        self.assertTrue(all(len(p['evaluation_combinations']) == 14 for p in plans))
        for job in jobs:
            command = job['command']
            self.assertEqual(job['dataset_type'], 'paired')
            self.assertIn('--nproc_per_node=8', command)
            self.assertEqual(command[command.index('--gradient-accumulation-steps') + 1], '1')
            self.assertEqual(command[command.index('--batch-size') + 1], '64')
            self.assertEqual(command[command.index('--epochs') + 1], '50')
            self.assertIn('--gradient-checkpointing', command)
        self.assertIn('--master_port=29621', jobs[0]['command'])
        self.assertIn('--master_port=29622', jobs[1]['command'])
        self.assertEqual(jobs[0]['initial_checkpoint'], jobs[1]['initial_checkpoint'])
        self.assertEqual(jobs[1]['command'][jobs[1]['command'].index('--gt-shuffle-prob') + 1], '0.0')
        self.assertEqual(jobs[1]['command'][jobs[1]['command'].index('--base-prob') + 1], '0.45')

    def test_detectors_use_cpu_and_wait_for_mh_after_available_data(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = lane_plan('detectors', 0, Path(folder))
        self.assertEqual([j['dataset_type'] for j in plan['training_jobs']], ['ph', 'paired', 'mh'])
        for job in plan['training_jobs']:
            self.assertEqual(job['stage'], 'detectors')
            command = job['command']
            self.assertEqual(command[command.index('--device') + 1], 'cpu')
            self.assertEqual(command[command.index('--num-workers') + 1], '2')


if __name__ == '__main__':
    unittest.main()
