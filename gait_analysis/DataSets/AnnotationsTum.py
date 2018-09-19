import os
from gait_analysis.utils.data_loading import not_NIF_frame_nums, load_sequence_annotation, remove_nif
import numpy as np
from torch.utils.data import Dataset
from gait_analysis.utils.files import format_data_path

class Annotations(Dataset):
    '''
    TumGAID_Dataset loader
    '''

    def __init__(self, dataset_items, tumgait_annotations_root, transform=None):
        '''
        :param tumgait_annotations_root:
        :param transform torch transform object
        '''
        # path manipulation:
        self.annotations_path = format_data_path(tumgait_annotations_root)
        self.dataset_items = dataset_items

    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self, idx):
        '''
        gets a Data Frame of valid annotations and correspondence indices in
        the scene
        :param idx:
        :return: annotation content as Dataframe
        '''

        dataset_item = self.dataset_items[idx]

        # load raw annotations
        annotations = self._load_annotation(dataset_item)

        # NIF stands for Not In Frame, IF stands for In Frame
        IF_indices = np.array(not_NIF_frame_nums(annotations))

        annotations = remove_nif(annotations, IF_indices)
        return annotations, IF_indices

    def _load_annotation(self, dataset_item):
        '''
        :param dataset_item:
        :return: annotations with NIF included (raw data).
        '''
        p_num, sequence = dataset_item
        annotation_file = os.path.join(self.annotations_path,
                                       'annotation_p{:03d}.ods'.format(p_num))
        df = load_sequence_annotation(annotation_file, sequence)
        return df


if __name__ == '__main__':
    pass
