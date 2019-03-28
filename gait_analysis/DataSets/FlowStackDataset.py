import torch
import logging
from torch.utils.data import Dataset
from gait_analysis import CasiaDataset, Composer, Config
from gait_analysis import IndexingCasia as Indexing
from statistics import mode

class FlowStackDataset(Dataset):
    def __init__(self , transform=None):
        config = Config.Config()
        self.config = config.config
        prev_config1 = self.config['indexing']['grouping']
        prev_config2 = self.config['transformers']

        self.config['indexing']['grouping'] = 'person_sequence'
        itemizer = Indexing()
        self.dataset_items = itemizer.get_items()
        # TODO: verification indexer in gropping people_sequence_angle mode
        # TODO: verify flows in output configuration even change it?
        # creating the data loader
        composer = Composer()
        transformer = composer.compose()
        self.config['indexing']['grouping'] = 'person_sequence_angle'
        self.config['transformers'] = {
            'AnnotationToLabel': {'target': ["annotations"]},
            'Transpose' : {'swapping': (2, 0, 1) , 'target': ["flows"]},
            'DimensionResize' : {'start':10,'dimension': 20, 'target': ["flows","annotations"]},
            'ToTensor': {'target':["flows","annotations"]}
        }
        self.dataset = CasiaDataset(transform=transformer)
        self.config['indexing']['grouping'] = prev_config1
        self.config['transformers'] = prev_config2
        self.transform = transform
        self.logger = logging.getLogger()

    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self , idx):

        # TODO READ FROM Config
        prev_config = self.config['dataset_output']
        self.config['dataset_output'] = {
            'data': ["flows"] ,
            'label': "annotations"
        }
        angles = self.config['flow']['angles']
        axis = self.config['flow']['axis']
        idx = self.dataset_items[idx]

        # item = [self.dataset[self.dataset.dataset_items.index((idx[0] , idx[1] , a))] for a in angles]
        item = [self.dataset[self.dataset.dataset_items.index((idx[0] , idx[1] , a))] for a in angles]
        # make annotation pooling
        annotations = tuple([item[i][1].reshape(1 , -1) for i in range(len(angles))])
        annotations = torch.cat(annotations , 0)
        annotations = torch.tensor([mode(annotations[: , i].numpy()) for i in range(annotations.size(1))])
        # make flow stack
        flows = []
        for i in range(annotations.size(0)):
            if axis == 'all':
                flow_timesteps = tuple([item[j][0]['flows'][1] for j in range(len(angles))])
            elif isinstance(axis,int) and axis < 3:
                flow_timesteps = [item[j][0]['flows'][1][1, :, :].unsqueeze(0) for j in range(len(angles))]
            else:
                logger = logging.getLogger()
                logger.error("Value on axis config['flow']['axis'] not supported")
                raise ValueError
            flow_timesteps = torch.cat(flow_timesteps , 2)
            flows.append(flow_timesteps)
        output = torch.cat(flows , 0).unsqueeze(0)
        if self.transform:
            output = self.transform(output)
        self.config['dataset_output'] = prev_config
        return output , annotations

