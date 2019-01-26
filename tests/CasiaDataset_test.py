import unittest
import cv2
from gait_analysis import CasiaDataset, settings
from gait_analysis.Config import Config
# import numpy as np
from gait_analysis import Composer
from gait_analysis import Rescale


def showImage(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class TestCasiaDataset(unittest.TestCase):
    def setUp(self):
        self.casia_options_dict = settings.tumgaid_default_args
        self.composer = Composer()
        self.config = Config()

    def test_len(self):
        datasets = CasiaDataset()
        self.assertEqual(415,len(datasets))

    def test_import(self):
        datasets = CasiaDataset()
        annotations = datasets[1]['annotations']
        self.assertEqual(62,len(annotations))
        # dataset_0 = dataset[0]
        # self.assertEqual(dataset_0.shape, (480, 640, 3))
    def test_rescaling(self):
        self.config.config['heatmaps']['load'] = True
        datasets = CasiaDataset()
        configutation = self.config.config
        item = datasets[0]
        ############
        # instatiatiate a transformer:
        ############
        parameters = configutation['transformers']['Rescale']
        rescale = Rescale(parameters) # this a rescaling
        transformed_item = rescale(item)






if __name__ == '__main__':
    unittest.main()
