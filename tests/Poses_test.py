import unittest
import cv2
from gait_analysis import Poses, settings


class TestAnnotations(unittest.TestCase):
    def setUp(self):
        self.tumgait_default_args = settings.tumgaid_default_args
        self.preprocessing_path = settings.tumgaid_preprocessing_root
        self.dataset_items = [(2, 'b01'),(300,'n02')]

    def test_len(self):
        poses = Poses(self.dataset_items, self.preprocessing_path, self.tumgait_default_args)
        self.assertEqual(2,len(poses))

    def test_import(self):
        poses = Poses(self.dataset_items, self.preprocessing_path, self.tumgait_default_args)
        pose = poses[1]
        self.assertEqual(2, len(pose))
        frame0 = pose[0]
        self.assertEqual(73, len(frame0))

if __name__ == '__main__':
    unittest.main()
