"""Independent experiment initialization, provenance and failure propagation."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from auto_pipeline import ExperimentRunner
from experiment_plan import build_plan


class PipelineTests(unittest.TestCase):
    def test_all_policy_jobs_start_from_same_initial_checkpoint(self):
        plan = build_plan(seeds=[0, 1, 2], num_gpus=2)
        policies = [job for job in plan['training_jobs'] if job['stage'] == 'policies']
        detectors = [job for job in plan['training_jobs'] if job['stage'] == 'detectors']
        self.assertEqual(len(policies), 12)
        self.assertEqual(len(detectors), 9)
        self.assertEqual(len(plan['evaluation_combinations']), 72)
        self.assertEqual(len({job['initial_checkpoint'] for job in policies}), 1)
        self.assertEqual({job['dataset_type'] for job in policies}, {'paired', 'mh'})
        for job in policies:
            command = job['command']
            self.assertEqual(command[command.index('--epochs') + 1], '50')
            self.assertEqual(command[command.index('--policy-mode') + 1], job['mode'])
            self.assertEqual(command[command.index('--precision') + 1], 'float32')
            self.assertEqual(command[command.index('--warmup-steps') + 1], '1000')
            self.assertEqual(command[command.index('--gradient-accumulation-steps') + 1], '1')
            self.assertEqual(command[command.index('--batch-size') + 1], '64')
            self.assertIn('--gradient-checkpointing', command)
            if job['mode'] == 'conditional':
                self.assertEqual(command[command.index('--gt-shuffle-prob') + 1], '0.0')
                self.assertEqual(command[command.index('--base-prob') + 1], '0.45')
        combinations = {(row['policy_dataset'], row['detector_dataset'])
                        for row in plan['evaluation_combinations'] if row['detector_dataset']}
        self.assertEqual(combinations, {('paired','ph'), ('paired','mh'), ('paired','paired'), ('mh','mh'), ('mh','ph')})
        for row in plan['evaluation_combinations']:
            self.assertEqual(row['recovery_enabled'], row['detector_checkpoint'] is not None)

    def test_success_records_and_reuses_exact_job_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = build_plan(root=root, output_root=root / 'outputs', seeds=[0], num_gpus=1)
            selected = plan['training_jobs'][:2]
            calls = []
            class Process:
                def __init__(self, command, **kwargs):
                    calls.append(command)
                    output = Path(command[command.index('--output') + 1])
                    (output / 'final_model.pt').touch()
                def wait(self): return 0
            runner = ExperimentRunner(plan, root / 'status.json')
            with patch.object(runner, 'ready'), patch('auto_pipeline.subprocess.Popen', Process):
                self.assertTrue(runner.run(job_ids=[job['id'] for job in selected]))
                self.assertTrue(runner.run(job_ids=[job['id'] for job in selected]))
            self.assertEqual(len(calls), 2)
            for command in calls:
                self.assertEqual(command[command.index('--checkpoint') + 1], selected[0]['initial_checkpoint'])

    def test_failure_stops_next_job(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = build_plan(root=root, output_root=root / 'outputs', seeds=[0], num_gpus=1)
            calls = []
            class Process:
                def __init__(self, command, **kwargs): calls.append(command)
                def wait(self): return 7
            runner = ExperimentRunner(plan, root / 'status.json')
            with patch.object(runner, 'ready'), patch('auto_pipeline.subprocess.Popen', Process):
                self.assertFalse(runner.run())
            self.assertEqual(len(calls), 1)
            self.assertEqual(runner.status['status'], 'failed')
            self.assertEqual(next(iter(runner.status['jobs'].values()))['status'], 'failed')


if __name__ == '__main__': unittest.main()
