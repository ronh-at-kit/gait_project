import unittest

from gait_analysis import Config
from gait_analysis import FlowStackDataset


class TestCasiaDataset(unittest.TestCase):
    def setUp(self):
        self.config = Config.Config()

    def test_len(self):
        datasets = FlowStackDataset()
        self.assertEqual(415,len(datasets))

    def test_import(self):
        datasets = FlowStackDataset()
        annotations = datasets[1]['annotations']
        self.assertEqual(62,len(annotations))
        # dataset_0 = dataset[0]
        # self.assertEqual(dataset_0.shape, (480, 640, 3))






if __name__ == '__main__':
    unittest.main()
