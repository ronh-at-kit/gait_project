import unittest
from gait_analysis import Annotations


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

        self.annotations_path = "~/Documents/TUMData/"

    def test_len(self):
        annotations = Annotations(self.annotations_path,self.tumgait_default_args);
        self.assertEqual(len(annotations), 66)

    def test_import(self):
        annotations = Annotations(self.annotations_path,self.tumgait_default_args);
        annotation = annotations[1]
        self.assertEqual((61, 3), annotation.shape)
        self.assertTrue((['frame_id','left_foot','right_foot']==annotation.columns).all())



if __name__ == '__main__':
    unittest.main()
