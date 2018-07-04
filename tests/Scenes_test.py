import unittest
import cv2
from gait_analysis import Scenes


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

        self.annotations_path = "~/Documents/TUMData/annotations"
        self.scenes_path = "~/Documents/TUMData/TUMGAIDimage"

    def test_len(self):
        scenes = Scenes(self.scenes_path, self.annotations_path, self.tumgait_default_args)
        self.assertEqual(66,len(scenes))

    def test_import(self):
        scenes = Scenes(self.scenes_path, self.annotations_path, self.tumgait_default_args)
        scene = scenes[1]
        self.assertEqual(len(scene), 61)
        scene_0 = scene[0]
        self.assertEqual(scene_0.shape, (480, 640, 3))
        # showImage(scene_0)



if __name__ == '__main__':
    unittest.main()
