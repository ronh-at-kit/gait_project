from torch.utils.data import Dataset
from gait_analysis import AnnotationsCasia as Annotations
from gait_analysis import PosesCasia as Poses
from gait_analysis import ScenesCasia as Scenes
from gait_analysis import FlowsCasia as Flows
from gait_analysis import IndexingCasia as Indexing
from gait_analysis import HeatMapsCasia as HeatMaps
from gait_analysis.Config import Config

class CasiaDataset(Dataset):
    # TODO: options_dict comes from a config reader.
    def __init__(self, transform=None):

        # TODO: use include scenes to filter out sequences.
        # list(product(person_numbers, options_dict['include_scenes']))
        # self.person_numbers = [item[1] for item in dataset_items]

        itemizer = Indexing();
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
        if self.config['heatmaps']['load']:
            self.heatmaps = HeatMaps(self.dataset_items)
        self.transform = transform


    def __len__(self):
        return len(self.dataset_items)

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
        if self.transform:
            output = self.transform(output)
        if 'dataset_output' in self.config:
            dataset_output = {}
            for k in self.config['dataset_output']['data']:
                dataset_output[k] = output[k]
            return dataset_output, output[self.config['dataset_output']['label']]
        else:
            return output



