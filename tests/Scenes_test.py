import unittest
import cv2
from gait_analysis import ScenesTum, settings
import numpy as np

def showImage(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class TestAnnotations(unittest.TestCase):
    def setUp(self):
        self.tumgait_default_args = settings.tumgaid_default_args
        self.dataset_items = [(2, 'b01'),(300,'n02')]
        self.scenes_path = settings.tumgaid_root

    def test_len(self):
        scenes = ScenesTum(self.dataset_items , self.scenes_path , self.tumgait_default_args)
        self.assertEqual(2,len(scenes))

    def test_import(self):
        scenes = ScenesTum(self.dataset_items , self.scenes_path , self.tumgait_default_args)
        scene = scenes[1]
        self.assertEqual(75,len(scene))
        scene_0 = scene[0]
        self.assertEqual(scene_0.shape, (480, 640, 3))
        #showImage(scene_0)
    def test_import_valid_in_frame(self):
        valid_indices = np.asarray([False] * 75)
        valid_indices[10:14] = True
        self.tumgait_default_args['valid_indices'] = valid_indices

        scenes = ScenesTum(self.dataset_items , self.scenes_path , self.tumgait_default_args)
        scene = scenes[1]
        self.assertEqual(4,len(scene))
        scene_0 = scene[0]
        self.assertEqual(scene_0.shape, (480, 640, 3))
        # showImage(scene_0)



if __name__ == '__main__':
    unittest.main()
