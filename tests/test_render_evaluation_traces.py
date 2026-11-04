from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from render_evaluation_traces import action_events,normalize_video_name,video_filename


class TraceRenderingTests(unittest.TestCase):
    def test_video_filename_marks_outcome(self):
        self.assertEqual(video_filename(7,True),'episode_007_success.mp4')
        self.assertEqual(video_filename(12,False),'episode_012_failed.mp4')

    def test_existing_video_is_renamed_without_reencoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            original=folder/'episode_003.mp4'
            original.write_bytes(b'video-data')
            renamed=normalize_video_name(folder,3,False)
            self.assertEqual(renamed.name,'episode_003_failed.mp4')
            self.assertEqual(renamed.read_bytes(),b'video-data')
            self.assertFalse(original.exists())

    def test_replay_actions_include_forward_and_retreat_steps(self):
        trace=[dict(step=1,phase='forward',action=[0.]*7),
               dict(step=1,phase='detection'),
               dict(step=2,phase='retreat',action=[1.]*7)]
        self.assertEqual(action_events(trace,2),[trace[0],trace[2]])

    def test_replay_rejects_missing_or_reordered_physical_steps(self):
        trace=[dict(step=1,phase='forward',action=[0.]*7),
               dict(step=3,phase='retreat',action=[1.]*7)]
        with self.assertRaisesRegex(ValueError,'physical step count'):
            action_events(trace,2)


if __name__=='__main__':unittest.main()
