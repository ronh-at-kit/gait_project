
import unittest
from gait_analysis import Composer


class TestCasiaDataset(unittest.TestCase):
    def setUp(self):
        self.composer = Composer()

    def test_compose(self):
        transformer = self.composer.compose()
        print(transformer)
