from torch.utils.data import Dataset
from gait_analysis import AnnotationsCasia as Annotations
from gait_analysis import CasiaItemizer as Itemizer

class CasiaDataset(Dataset):
    # TODO: options_dict comes from a config reader.
    def __init__(self, options_dict):

        # TODO: use include scenes to filter out subsequences.
        # list(product(person_numbers, options_dict['include_scenes']))
        itemizer = Itemizer('person-sequence');
        self.dataset_items = itemizer.get_items()
        self.options_dict = options_dict # this will come from a config file
        # self.person_numbers = [item[1] for item in dataset_items]
        self.annotations = Annotations(self.dataset_items)
    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self, idx):
        annotations, in_frame_indices = self.annotations[idx]
        return annotations



