import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_evaluate_mh_baseline_2k as pipeline


class MHBaselinePipelineTests(unittest.TestCase):
    def test_failed_training_never_starts_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = pipeline.Pipeline(root, root / 'evaluation', root / 'catalog.json', list('01234567'))
            with patch.object(runner, 'execute', side_effect=RuntimeError('training failed')) as execute, \
                    patch.object(pipeline, 'verify_checkpoint') as verify:
                with self.assertRaisesRegex(RuntimeError, 'training failed'):
                    runner.run()
            execute.assert_called_once()
            verify.assert_not_called()
            self.assertEqual(json.loads((root / 'pipeline_status.json').read_text())['status'], 'failed')

    def test_wrong_checkpoint_never_starts_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = pipeline.Pipeline(root, root / 'evaluation', root / 'catalog.json', list('01234567'))
            with patch.object(runner, 'execute') as execute, \
                    patch.object(pipeline, 'verify_checkpoint', side_effect=ValueError('wrong recipe')):
                with self.assertRaisesRegex(ValueError, 'wrong recipe'):
                    runner.run()
            self.assertEqual(execute.call_count, 1)

    def test_successful_training_hands_same_checkpoint_to_eight_gpu_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = pipeline.Pipeline(root, root / 'evaluation', root / 'catalog.json', list('01234567'))
            checkpoint = dict(path=str(root / 'checkpoints/step_002000.pt'), step=2000)
            summary = dict(successes=35, episodes=50)
            with patch.object(runner, 'execute') as execute, \
                    patch.object(pipeline, 'verify_checkpoint', return_value=checkpoint), \
                    patch('evaluate_can.checkpoint_digest', return_value='sha256'), \
                    patch.object(pipeline, 'audit_evaluation', return_value=summary) as audit:
                runner.run()
            self.assertEqual(execute.call_count, 2)
            command = execute.call_args_list[1].args[1]
            self.assertEqual(command[command.index('--gpus') + 1:command.index('--max-retries')], list('01234567'))
            self.assertEqual(command[command.index('--checkpoint') + 1], checkpoint['path'])
            self.assertNotIn('--detector', command)
            audit.assert_called_once_with(root / 'evaluation', root / 'catalog.json', checkpoint, 'sha256')
            self.assertEqual(runner.state['status'], 'completed')

    def test_checkpoint_verification_rejects_partial_conditional_and_resumed_runs(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'checkpoints').mkdir()
            (root / 'final_model.pt').touch()
            args = dict(dataset_type='mh', policy_mode='baseline', num_steps=2000, lr_decay_steps=2000,
                        batch_size=64, seed=0, supervised_action_dims=7,
                        training_recipe='pi05_can_full_sft_physical7_v4', resume=None)
            payload = dict(step=2000, args=args, rank_rng=[None] * 8, model_state_dict={}, normalization_stats={})
            path = root / 'checkpoints/step_002000.pt'
            torch.save(payload, path)
            self.assertEqual(pipeline.verify_checkpoint(root)['step'], 2000)
            for field, value in [('policy_mode', 'conditional'), ('num_steps', 1000), ('resume', 'old.pt')]:
                changed = dict(payload, args=dict(args, **{field: value}))
                torch.save(changed, path)
                with self.assertRaises(ValueError):
                    pipeline.verify_checkpoint(root)
            torch.save(dict(payload, step=1500), path)
            with self.assertRaises(ValueError):
                pipeline.verify_checkpoint(root)
            torch.save(dict(payload, rank_rng=[None]), path)
            with self.assertRaises(ValueError):
                pipeline.verify_checkpoint(root)
            torch.save(dict(payload, model_state_dict={'action_condition_proj.weight': torch.zeros(1)}), path)
            with self.assertRaises(ValueError):
                pipeline.verify_checkpoint(root)


if __name__ == '__main__':
    unittest.main()
