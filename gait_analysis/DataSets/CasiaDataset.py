from torch.utils.data import Dataset
from gait_analysis import AnnotationsCasia as Annotations
from gait_analysis import PosesCasia as Poses
from gait_analysis import ScenesCasia as Scenes
from gait_analysis import FlowsCasia as Flows
from gait_analysis import CropsFlowCasia as CropsFlow
from gait_analysis import IndexingCasia as Indexing
from gait_analysis import HeatMapsCasia as HeatMaps
from gait_analysis.Config import Config
from gait_analysis.utils import training
# from memory_profiler import profile

import logging
class CasiaDataset(Dataset):
    def __init__(self, transform=None):
        itemizer = Indexing()
        self.dataset_items = itemizer.get_items()
        config = Config()
        self.config = config.config
        self.annotations = Annotations(self.dataset_items)
        if self.config['pose']['load']:
            self.poses = Poses(self.dataset_items)
        if self.config['scenes']['load']:
            self.scenes = Scenes(self.dataset_items)
        if self.config['flow']['load']:
            self.flows = Flows(self.dataset_items)
        if self.config['crops_flow']['load']:
            self.crops_flow = CropsFlow(self.dataset_items)
        if self.config['heatmaps']['load']:
            self.heatmaps = HeatMaps(self.dataset_items)
        self.transform = transform
        self.logger = logging.getLogger()


    def __len__(self):
        return len(self.dataset_items)
    # @profile()
    def __getitem__(self, idx):
        output = {}
        # annotations are always in the output:
        annotations, in_frame_indices = self.annotations[idx]
        output['annotations'] = annotations
        # adding all other optional features
        if hasattr(self, 'poses'):
            self.poses.set_option('valid_indices',in_frame_indices)
            poses, valid_poses = self.poses[idx]
            output['poses'] = poses
            in_frame_indices = valid_poses
        if hasattr(self,'scenes'):
            self.scenes.set_option('valid_indices' , in_frame_indices)
            output['scenes'] = self.scenes[idx]
        if hasattr(self , 'flows'):
            self.flows.set_option('valid_indices' , in_frame_indices)
            output['flows'] = self.flows[idx]
        if hasattr(self, 'heatmaps'):
            self.heatmaps.set_option('valid_indices', in_frame_indices)
            output['heatmaps'] = self.heatmaps[idx]
        if hasattr(self, 'crops_flow'):
            self.crops_flow.set_option('valid_indices', in_frame_indices)
            output['crops_flow'] = self.crops_flow[idx]
        if self.transform:
            output = self.transform(output)

        if 'dataset_output' in self.config:
            dataset_output = {}
            for k in self.config['dataset_output']['data']:
                dataset_output[k] = output[k]
            labels_output = output[self.config['dataset_output']['label']]
            # del output
            # del annotations
            # self.logger.info('Memory stats inside the Casia Dataset')
            # training.print_memory_size(locals(), self.logger)
            return dataset_output, labels_output
        else:
            return output



