from unittest import TestCase
from gait_analysis.Config import Config
from gait_analysis.settings import configuration

class TestConfig(TestCase):
    def test___init__(self):
        configuration = 'flows'
        c = Config()
