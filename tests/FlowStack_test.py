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
        item = datasets[0]
        print(item)



if __name__ == '__main__':
    unittest.main()
