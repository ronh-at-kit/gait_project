import unittest
import cv2
from gait_analysis import Scenes
import numpy as np

def showImage(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

        self.dataset_items = [(2, 'b01'),(300,'n02')]
        self.scenes_path = "~/Documents/TUMData/TUMGAIDimage"

    def test_len(self):
        scenes = Scenes(self.dataset_items, self.scenes_path, self.tumgait_default_args)
        self.assertEqual(2,len(scenes))

    def test_import(self):
        scenes = Scenes(self.dataset_items, self.scenes_path, self.tumgait_default_args)
        scene = scenes[1]
        self.assertEqual(75,len(scene))
        scene_0 = scene[0]
        self.assertEqual(scene_0.shape, (480, 640, 3))
        #showImage(scene_0)
    def test_import_valid_in_frame(self):
        valid_indices = np.asarray([False] * 75)
        valid_indices[10:14] = True
        self.tumgait_default_args['valid_indices'] = valid_indices

        scenes = Scenes(self.dataset_items, self.scenes_path, self.tumgait_default_args)
        scene = scenes[1]
        self.assertEqual(4,len(scene))
        scene_0 = scene[0]
        self.assertEqual(scene_0.shape, (480, 640, 3))
        # showImage(scene_0)



if __name__ == '__main__':
    unittest.main()
