import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_can import select_episodes
from evaluate_checkpoint_shards import merge_results, shard_ranges, worker_command
import evaluate_checkpoint_shards as shards


class EvaluationShardTests(unittest.TestCase):
    def test_failed_shard_retries_without_restarting_successful_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / 'policy.pt'
            checkpoint.touch()
            output = root / 'results'
            failed = mock.Mock(pid=101, returncode=1)
            failed.poll.return_value = 1
            successful = mock.Mock(pid=102, returncode=0)
            successful.poll.return_value = 0
            retried = mock.Mock(pid=103, returncode=0)
            retried.poll.return_value = 0
            arguments = ['evaluate_checkpoint_shards.py', '--checkpoint', str(checkpoint),
                         '--output', str(output), '--gpus', '0', '1', '--max-retries', '1']
            with mock.patch.object(sys, 'argv', arguments), \
                    mock.patch.object(shards.subprocess, 'Popen', side_effect=[failed, successful, retried]) as popen, \
                    mock.patch.object(shards.os, 'killpg') as killpg, \
                    mock.patch.object(shards, 'merge_results', return_value={'success_rate': .5}):
                shards.main()
            self.assertEqual(popen.call_count, 3)
            self.assertEqual(popen.call_args_list[0].args[0], popen.call_args_list[2].args[0])
            killpg.assert_called_once_with(101, shards.signal.SIGTERM)
            state = json.loads((output / 'status.json').read_text())
            self.assertEqual(state['status'], 'completed')
            self.assertEqual([worker['attempts'] for worker in state['workers']], [2, 1])

    def test_recovery_arguments_preserve_shard_and_zero_scale_detector(self):
        for scale in (0., .2, 2.5, 4.5):
            command = worker_command('policy.pt', 'catalog.json', 'output', 25, 50, 'ph.pt', scale,
                                     negative_prompt_mode='repeat_last_action')
            self.assertEqual(command[command.index('--detector') + 1], 'ph.pt')
            self.assertEqual(float(command[command.index('--scale') + 1]), scale)
            self.assertEqual(command[command.index('--episode-start') + 1], '25')
            self.assertEqual(command[command.index('--episode-end') + 1], '50')
            self.assertEqual(command[command.index('--negative-prompt-mode') + 1], 'repeat_last_action')
            self.assertIn('--resume', command)
        self.assertNotIn('--detector', worker_command('policy.pt', 'catalog.json', 'output', 0, 25))

    def test_balanced_ranges_preserve_full_catalog(self):
        self.assertEqual(shard_ranges(50, 3), [(0, 17), (17, 34), (34, 50)])
        self.assertEqual(shard_ranges(50, 5), [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)])
        for workers in range(1, 51):
            ranges = shard_ranges(50, workers)
            self.assertEqual([index for start, end in ranges for index in range(start, end)], list(range(50)))
            sizes = [end - start for start, end in ranges]
            self.assertLessEqual(max(sizes) - min(sizes), 1)
        for workers in (0, 51):
            with self.assertRaises(ValueError):
                shard_ranges(50, workers)

    def test_global_indices_and_seeds_are_preserved(self):
        catalog = {'episodes': [{'id': i, 'seed': 100000 + i} for i in range(50)]}
        selected = [item for start in range(0, 50, 10) for item in select_episodes(catalog, 50, start, start + 10)]
        self.assertEqual([i for i, _ in selected], list(range(50)))
        self.assertEqual([item['seed'] for _, item in selected], list(range(100000, 100050)))
        for start, end in [(-1, 10), (10, 10), (40, 51)]:
            with self.assertRaises(ValueError):
                select_episodes(catalog, 50, start, end)

    def fixture(self, root):
        episodes = [{'id': i, 'seed': 100000 + i} for i in range(50)]
        catalog = root / 'catalog.json'
        catalog.write_text(json.dumps({'episodes': episodes}))
        folders = []
        for start in range(0, 50, 10):
            folder = root / str(start)
            folder.mkdir()
            run = dict(episode_range=[start, start + 10], smoke=False, rollouts=50,
                       checkpoint='step8000.pt', checkpoint_step=8000, protocol={'horizon': 400},
                       checkpoint_sha256='same-weights', catalog_sha256='same-catalog',
                       fingerprint=str(start), policy_cuda_visible_devices=str(start // 10))
            (folder / 'run.json').write_text(json.dumps(run))
            rows = [dict(episode_id=i, environment_seed=100000 + i, success=i % 2 == 0,
                         physical_steps=200, recoveries=0) for i in range(start, start + 10)]
            (folder / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            (folder / 'summary.json').write_text(json.dumps(dict(episodes=10, successes=5)))
            folders.append(folder)
        return catalog, folders

    def test_full_merge_and_missing_shard_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, folders = self.fixture(root)
            result = merge_results(folders, catalog, 50, root)
            self.assertEqual(result['episodes'], 50)
            self.assertEqual(result['success_rate'], .5)
            self.assertTrue(result['complete_evaluation'])
            with self.assertRaises(ValueError):
                merge_results(folders[:-1], catalog, 50, root)

    def test_changed_checkpoint_or_duplicate_episode_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, folders = self.fixture(root)
            run_path = folders[-1] / 'run.json'
            run = json.loads(run_path.read_text())
            run['checkpoint_sha256'] = 'different-weights'
            run_path.write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, 'differs'):
                merge_results(folders, catalog, 50, root)
            run['checkpoint_sha256'] = 'same-weights'
            run_path.write_text(json.dumps(run))
            records = folders[-1] / 'episodes.jsonl'
            records.write_text(records.read_text() + records.read_text().splitlines()[0] + '\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                merge_results(folders, catalog, 50, root)

    def test_conditioning_ablation_cannot_merge_with_normal_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog, folders = self.fixture(root)
            for index, folder in enumerate(folders):
                path = folder / 'run.json'
                run = json.loads(path.read_text())
                run['action_conditioning_enabled'] = index != 0
                path.write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, 'differs'):
                merge_results(folders, catalog, 50, root)


if __name__ == '__main__':
    unittest.main()
