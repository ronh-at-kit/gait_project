from gait_analysis import CasiaDataset, settings
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from gait_analysis.Config import Config

c = Config()
c.config['indexing']['grouping'] = 'person_sequence_angle'
c.config['pose']['load'] = True
c.config['flow']['load'] = True

datasets = CasiaDataset()
print(datasets.dataset_items[7])
item = datasets[7]