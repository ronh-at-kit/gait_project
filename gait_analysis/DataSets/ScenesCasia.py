import os
from memory_profiler import profile

import cv2
from os import listdir
from os.path import join
from torch.utils.data import Dataset

from gait_analysis.utils.files import format_data_path
from gait_analysis.utils.data_loading import read_image, extract_pnum
import gait_analysis.settings as settings
from gait_analysis.Config import Config



class ScenesCasia(Dataset):
    '''
    TumGAID_Dataset loader
    '''
    __slots__ = 'config' , 'images_dir', 'dataset_items', 'options'
    def __init__(self, dataset_items, transform=None):

        self.config = Config()
        if self.config.config['scenes']['crops']:
            self.images_dir = format_data_path(settings.casia_crops_dir)
        else:
            self.images_dir = format_data_path(settings.casia_images_dir)
        self.dataset_items = dataset_items
        self.options = {}


    def __len__(self):
        return len(self.dataset_items)
    # @profile()
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
        images = self.load_scene(dataset_item)
        return images

    def set_option(self,key, value):
        self.options[key] = value
    # @profile()
    def load_scene(self , dataset_item):
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

        scene_folder = os.path.join(self.images_dir,'{:03d}'.format(p_num),sequence)
        def compose_image_filename(angle,i):
            return join(scene_folder , '{}-{:03d}'.format(sequence , angle) , \
                    '{:03d}-{}-{:03d}_frame_{:03d}.jpg'.format(p_num , sequence , angle , i))

        if 'valid_indices' in self.options:
            valid_indices = self.options['valid_indices']
            scene_files = [compose_image_filename(angle , i) for i , valid in enumerate(valid_indices) if valid]
        else:
            scene_files = [join(scene_folder, '{}-{:03d}'.format(sequence , angle) , f) for f in listdir(scene_folder) if f.endswith('.jpg')]
            scene_files.sort()
        scene_images = []
        for f in scene_files:
            image = read_image(f)
            scene_images.append(image)
        return scene_images



if __name__ == '__main__':
    pass