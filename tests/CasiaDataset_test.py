import unittest
import cv2
from gait_analysis import CasiaDataset, settings
import numpy as np

def showImage(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class TestCasiaDataset(unittest.TestCase):
    def setUp(self):
        self.casia_options_dict = settings.tumgaid_default_args
        self.images_dir = settings.CASIA_IMAGES_DIR
        self.preprocessing_dir = settings.CASIA_PREPROCESSING_DIR
        self.annotations_dir = settings.CASIA_ANNOTATIONS_DIR

    def test_len(self):
        datasets = CasiaDataset(self.images_dir, self.preprocessing_dir, self.annotations_dir, self.casia_options_dict)
        self.assertEqual(66,len(datasets))

    def test_import(self):
        datasets = CasiaDataset(self.images_dir, self.preprocessing_dir, self.annotations_dir, self.casia_options_dict)
        dataset = datasets[1]
        self.assertEqual(75,len(dataset))
        dataset_0 = dataset[0]
        # self.assertEqual(dataset_0.shape, (480, 640, 3))



if __name__ == '__main__':
    unittest.main()
