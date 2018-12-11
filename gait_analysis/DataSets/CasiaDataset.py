from torch.utils.data import Dataset
from gait_analysis import Scenes, Poses
from gait_analysis import AnnotationsCasia as Annotations
from gait_analysis.utils.files import format_data_path
from gait_analysis.utils.data_loading import list_annotations_files
import gait_analysis.settings as settings
import os


class CasiaDataset(Dataset):
    # TODO: options_dict comes from a config reader.
    def __init__(self, options_dict):
        annotation_files = list_annotations_files(settings.casia_annotations_dir)
        # TODO: dataset_items comes from a itemizer class
        dataset_items = list(map(extract_pnum_subsequence, annotation_files))
        # TODO: use include scenes to filter out subsequences.
        # list(product(person_numbers, options_dict['include_scenes']))
        self.dataset_items = dataset_items
        self.options_dict = options_dict # this will come from a config file
        # self.person_numbers = [item[1] for item in dataset_items]
        self.annotations = Annotations(self.dataset_items)



    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self, idx):
        annotations, in_frame_indices = self.annotations[idx]
        return annotations


def extract_pnum_subsequence(abspath):
    filename = abspath.split(os.path.sep)[-1]
    p_num = filename[0:3]
    subsequence = filename[4:9]
    return (int(p_num), subsequence)
