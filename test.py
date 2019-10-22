import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import os.path as path
import copy
import matplotlib.pyplot as plt
# from torch.utils.data.sampler import SequentialSampler


from gait_analysis import AnnotationsCasia as Annotations
from gait_analysis import CasiaDataset
from gait_analysis.Config import Config
from gait_analysis import Composer

import matplotlib.pyplot as plt

print("done")

TIMESTEPS = 40 # size videos
BATCH_SIZE = 2 #until now just batch_size = 1

#change configuration in settings.py
c = Config()
c.config['indexing']['grouping'] = 'person_sequence_angle'
c.config['transformers']['DimensionResize']['dimension'] = TIMESTEPS
#c.config['transformers']= {
        #'Transpose' : {'swapping': (2,1,0) , 'target': ["scenes"]},
       # 'Rescale': {'output_size' : (240,320), 'target': ["scenes"]},
        #'AnnotationToLabel': {'target': ["annotations"]},
        #'Transpose' : {'swapping': (1,2,0) , 'target': ["heatmaps"]},
        #'Rescale': {'output_size' : (240,320), 'target': ["heatmaps"]},
        #'DimensionResize' : {'start': 0, 'dimension': 10, 'target': ["heatmaps","annotations"]},
        #'ToTensor': {'target':["heatmaps","annotations"]}
 #   }

print(c.config)
composer = Composer()
transformer = composer.compose()
dataset = CasiaDataset(transform=transformer)
print("Desired configuration: heatmaps_1")

data,_ =dataset[3]

print(len(dataset))

img1 = data["scenes"]
poses1 = data["poses"]

print(len(img1))
print(poses1)
print(poses1.size())