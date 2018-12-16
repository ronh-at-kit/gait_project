import os
import cv2
from os import listdir
from os.path import join
from torch.utils.data import Dataset

from gait_analysis.utils.files import format_data_path
import gait_analysis.settings as settings
from gait_analysis.Config import Config



class FlowsCasia(Dataset):
    '''
    TumGAID_Dataset loader
    '''

    def __init__(self, dataset_items, transform=None):
        self.flow_dir = format_data_path(settings.casia_flow_dir)
        self.dataset_items = dataset_items
        self.config = Config()
        self.options = {}


    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self, idx):
        '''
        returns a a dictionary with the outputs that were configured int the constructor
        and
        annotations
        :param idx:
        :return:
        '''

        dataset_item = self.dataset_items[idx]
        output = {}

        # Loading scene images
        images = self._load_flow(dataset_item)
        return images

    def set_option(self,key, value):
        self.options[key] = value

    def _load_flow(self , dataset_item):
        '''
        Loads scene sequence given a dataset item.
        :param dataset_item: the tuple indicating the dataset item
        :param not_nif_frames: a boolean mask indicating the valid frames. True means valid
        :return: a list of np.arrays containing the scene images
        '''
        grouping = self.config.get_indexing_grouping()
        if grouping == 'person_sequence':
            p_num, sequence = dataset_item
            # by default we select the 90 degree grom person_sequence grouping
            angle = 90
        elif grouping == 'person_sequence_angle':
            p_num , sequence, angle = dataset_item

        flow_folder = os.path.join(self.flow_dir , '{:03d}'.format(p_num) , sequence)
        def compose_flow_filename(angle,i):
            # 001_bg-01-018_frame_019_flow.png

            return join(flow_folder , '{}-{:03d}'.format(sequence , angle) , \
                    '{:03d}-{}-{:03d}_frame_{:03d}_flow.png'.format(p_num , sequence , angle , i))

        scene_files = []
        if 'valid_indices' in self.options:
            valid_indices = self.options['valid_indices']
            scene_files += [compose_flow_filename(angle , i) for i , valid in enumerate(valid_indices) if valid]
        else:
            scene_files += [join(flow_folder, '{}-{:03d}'.format(sequence , angle) , f) for f in listdir(flow_folder) if f.endswith('.png')]
            scene_files.sort()

        def read_image(im_file):
            if not os.path.isfile(im_file):
                raise ValueError('{} don\'t exist.'.format(im_file))
            im = cv2.imread(im_file, -1)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            return im
        scene_images = [read_image(f) for f in scene_files]
        return scene_images


def extract_pnum(abspath):
    path = os.path.basename(abspath)
    p_num = path[-7:-4]
    return int(p_num)


if __name__ == '__main__':
    pass