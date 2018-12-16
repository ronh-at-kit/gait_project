import unittest
import cv2
from gait_analysis import CasiaDataset, settings
# import numpy as np

def showImage(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class TestCasiaDataset(unittest.TestCase):
    def setUp(self):
        self.casia_options_dict = settings.tumgaid_default_args

    def test_len(self):
        datasets = CasiaDataset()
        self.assertEqual(415,len(datasets))

    def test_import(self):
        datasets = CasiaDataset()
        annotations = datasets[1]['annotations']
        self.assertEqual(62,len(annotations))
        # dataset_0 = dataset[0]
        # self.assertEqual(dataset_0.shape, (480, 640, 3))



if __name__ == '__main__':
    unittest.main()
