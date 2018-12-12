from torch.utils.data import Dataset
from gait_analysis import AnnotationsCasia as Annotations
from gait_analysis import PosesCasia as Poses
from gait_analysis import CasiaItemizer as Itemizer
from gait_analysis.Config import Config

class CasiaDataset(Dataset):
    # TODO: options_dict comes from a config reader.
    def __init__(self, options_dict):

        # TODO: use include scenes to filter out subsequences.
        # list(product(person_numbers, options_dict['include_scenes']))
        # self.person_numbers = [item[1] for item in dataset_items]

        itemizer = Itemizer('person-sequence');
        self.dataset_items = itemizer.get_items()
        config = Config()
        self.config = config.config
        self.annotations = Annotations(self.dataset_items)
        if self.config['pose']['load']:
            self.poses = Poses(self.dataset_items)

    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self, idx):
        output = {}
        # annotations are always in the output:
        annotations, in_frame_indices = self.annotations[idx]
        output['annotations'] = annotations
        # adding all other optional features
        if self.poses:
            self.poses.set_option('valid_indices',in_frame_indices)
            output['poses'] = self.poses[idx]
        return output



