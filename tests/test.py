import gc
import os
import psutil
from torch.utils.data import Dataset

from gait_analysis import CasiaDataset, ScenesCasia,Composer, Config
from gait_analysis.utils import training
from gait_analysis.utils.files import set_logger
c = Config.Config()
time_stamp = training.get_time_stamp()

logger, log_folder =  set_logger('TEST',c,time_stamp=time_stamp)
class DatasetContainer(Dataset):
    def __init__(self,dataset_items):
        self.images = ScenesCasia(dataset_items)
        self.dataset_items = dataset_items
    def __len__(self):
        return len(self.dataset_items)
    def __getitem__(self,idx):
        return self.images[idx]

def get_dataset():
    composer = Composer()
    transformer = composer.compose()
    casiaDataset = CasiaDataset(transform=transformer)
    # return DatasetContainer(casiaDataset.dataset_items)
    return casiaDataset


def funtion_to_train():
    # instantiates dataset
    dataset = get_dataset()
    pid = os.getpid()
    logger.info('Process running in PID: {}'.format(pid))
    py = psutil.Process(pid)
    it = 0
    prev_mem = py.memory_info()[0] / 2. ** 20
    stop_val = len(dataset)
    for i in range(stop_val):
        d = dataset[i]
        gc.collect()
        it+=1
        cur_mem = py.memory_info()[0] / 2. ** 20
        add_mem = cur_mem - prev_mem
        logger.info(" ===> MEMORY iteration{}: {}M +{}M  ==>>> {}M".format(it ,prev_mem, add_mem, cur_mem))
        prev_mem = cur_mem

