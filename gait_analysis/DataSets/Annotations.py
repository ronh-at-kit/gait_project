import os
import glob
from itertools import product
from gait_analysis.utils.data_loading import not_NIF_frame_nums, load_sequence_annotation, remove_nif
import numpy as np
from torch.utils.data import Dataset



class Annotations(Dataset):
    '''
    TumGAID_Dataset loader
    '''

    def __init__(self, tumgait_annotations_root, args_dict, transform=None):
        '''
        :param tumgait_annotations_root:
        :param args_dict: options
        :param transform torch transform object
        '''
        # path manipulation:
        if tumgait_annotations_root.startswith("~"):
            tumgait_annotations_root = os.path.expanduser(tumgait_annotations_root)
        if not os.path.exists(tumgait_annotations_root):
            raise ValueError('tumgait_annotation_root don\'t exist.')

        self.tumgaid_annotations_root = tumgait_annotations_root
        annotation_files = sorted(
            glob.glob(
                os.path.join(
                    tumgait_annotations_root, 'annotation_p*.ods'
                )
            )
        )
        p_nums = map(extract_patient_number, annotation_files)
        all_options = list(product(p_nums, args_dict['include_scenes']))

        self.p_nums = p_nums
        self.dataset_items = all_options
        self.options_dict = args_dict


    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self, idx):
        '''
        returns a a dictionary with the outputs that were configured int the constructor
        and
        annotations
        :param idx:
        :return: annotation content as Dataframe
        '''

        dataset_item = self.dataset_items[idx]

        # returns annotations with NIF
        annotations = self._load_annotation(dataset_item)
        nif_pos = not_NIF_frame_nums(annotations)
        nif_pos = np.array(nif_pos)

        annotations = remove_nif(annotations, nif_pos)
        # use NIF positions to filter out scene images, flow_maps, and others
        return annotations

    def _load_annotation(self, dataset_item):
        '''
        :param dataset_item:
        :return: annotations with NIF included.
        '''
        p_num, sequence = dataset_item
        annotation_file = os.path.join(self.tumgaid_annotations_root,
                                       'annotation_p{:03d}.ods'.format(p_num))
        df = load_sequence_annotation(annotation_file, sequence)
        return df


def extract_patient_number(abspath):
    path = os.path.basename(abspath)
    p_num = path[-7:-4]
    return int(p_num)


if __name__ == '__main__':
    pass