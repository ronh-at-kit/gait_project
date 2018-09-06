import unittest
import cv2
from gait_analysis import TumGAID_Dataset
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

        self.tumgait_root = "~/Documents/TUMData/TUMGAIDimage"
        self.preprocessing_root = "~/Documents/TUMData/preprocessing"
        self.annotations_root = "~/Documents/TUMData/annotations"

    def test_len(self):
        datasets = TumGAID_Dataset(self.tumgait_root, self.preprocessing_root,self.annotations_root, self.tumgait_default_args)
        self.assertEqual(66,len(datasets))

    def test_import(self):
        datasets = TumGAID_Dataset(self.tumgait_root, self.preprocessing_root,self.annotations_root, self.tumgait_default_args)
        dataset = datasets[1]
        self.assertEqual(75,len(dataset))
        dataset_0 = dataset[0]
        # self.assertEqual(dataset_0.shape, (480, 640, 3))



if __name__ == '__main__':
    unittest.main()
