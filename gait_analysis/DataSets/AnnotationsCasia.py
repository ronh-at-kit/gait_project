import os
import numpy as np
import gait_analysis.settings as settings
from gait_analysis.utils.data_loading import not_NIF_frame_nums, load_sequence_angle_annotation, remove_nif
from torch.utils.data import Dataset
from gait_analysis.utils.files import format_data_path
from gait_analysis.Config import Config



class AnnotationsCasia(Dataset):
    '''
    TumGAID_Dataset loader
    '''

    def __init__(self, dataset_items, transform=None):
        '''
        :param casia_annotations_root:
        :param transform torch transform object
        '''
        # path manipulation:
        self.annotations_path = format_data_path(settings.casia_annotations_dir)
        self.dataset_items = dataset_items
        self.config = Config()

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

        grouping = self.config.get_indexing_grouping()
        if grouping == 'person_sequence_angle':
            p_num , sequence, angle = dataset_item
        elif grouping == 'person_sequence':
            # +dev: this is hardcoded need to be define how we want the data.
            # angle = 90
            p_num , sequence = dataset_item
            angle = 90

        # load raw annotations
        annotations = self._load_annotation(p_num , sequence, angle)
        # NIF stands for Not In Frame, IF stands for In Frame


        IF_indices = np.array(not_NIF_frame_nums(annotations))
        annotations = remove_nif(annotations, IF_indices)
        return annotations, IF_indices

    def _load_annotation(self , person , sequence, angle):

        '''
                        :param person:
                        :param sequence:
                        :return: annotations with NIF included (raw data).
                        '''
        person = '{:03d}'.format(person)
        annotation_file = os.path.join(self.annotations_path, person, sequence,
                                       person + '-' + sequence + '-semiautomatic.ods')
        df = load_sequence_angle_annotation(annotation_file , sequence , angle)
        return df


if __name__ == '__main__':
    pass
