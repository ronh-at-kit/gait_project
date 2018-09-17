import unittest
import cv2
from gait_analysis import TumGAID_Dataset, settings
import numpy as np

def showImage(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class TestAnnotations(unittest.TestCase):
    def setUp(self):
        self.tumgait_default_args = settings.tumgaid_default_args
        self.tumgait_root = settings.tumgaid_root
        self.preprocessing_root = settings.tumgaid_preprocessing_root
        self.annotations_root = settings.tumgaid_annotations_root

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
