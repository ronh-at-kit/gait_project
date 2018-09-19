from gait_analysis import Scenes, Poses
from gait_analysis import AnnotationsCasia as Annotations
from torch.utils.data import Dataset
from gait_analysis.utils.files import format_data_path
from gait_analysis.utils.data_loading import list_annotations_files
from itertools import product
import os


class CasiaDataset(Dataset):
    def __init__(self, images_dir, preprocessing_dir, annotations_dir, options_dict):
        self.images_dir = format_data_path(images_dir)
        # TODO: preprocessing is not generated jet
        # self.preprocessing_dir = format_data_path(preprocessing_dir)
        self.annotations_dir = format_data_path(annotations_dir)

        annotation_files = list_annotations_files(self.annotations_dir)
        dataset_items = map(extract_pnum_subsequence, annotation_files)
        # TODO: use include scenes to filter out subsequences.
        # list(product(person_numbers, options_dict['include_scenes']))
        self.dataset_items = dataset_items
        self.options_dict = options_dict
        self.person_numbers = [item[1] for item in dataset_items]
        self.annotations = Annotations(self.dataset_items, self.annotations_dir)


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
