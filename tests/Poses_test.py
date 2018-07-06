import unittest
import cv2
from gait_analysis import Poses


class TestAnnotations(unittest.TestCase):
    def setUp(self):
        self.tumgait_default_args = {
            'load_pose': True,
            'load_pose_options': {
                'D': 2,
                'body_keypoints_include_list': ['LAnkle',
                                                'RAnkle',
                                                'LKnee',
                                                'RKnee',
                                                'RHip',
                                                'LHip']
            },
            'load_flow': True,
            'load_flow_options': {
                'method': 'dense',
                'load_patches': True,
                'load_patch_options': {
                    'patch_size': 5
                }
            },
            'load_scene': False,
            'load_scene_options': {
                'grayscale': False,
                'load_tracked': False
            },
            'include_scenes': ['b01', 'b02', 'n01', 'n02', 's01', 's02'],


        }

        self.annotations_path = "~/Documents/TUMData/annotations"
        self.preprocessing_path = "~/Documents/TUMData/preprocessing"

    def test_len(self):
        poses = Poses(self.preprocessing_path, self.annotations_path, self.tumgait_default_args)
        self.assertEqual(66,len(poses))

    def test_import(self):
        poses = Poses(self.preprocessing_path, self.annotations_path, self.tumgait_default_args)
        pose = poses[1]
        self.assertEqual(len(pose), 60)
        frame0 = pose[0]
        self.assertEqual(len(frame0), 6)

if __name__ == '__main__':
    unittest.main()
