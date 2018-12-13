import os
import cv2
from os import listdir
from os.path import join
from torch.utils.data import Dataset

from gait_analysis.utils.files import format_data_path
import gait_analysis.settings as settings
from gait_analysis.Config import Config



class ScenesCasia(Dataset):
    '''
    TumGAID_Dataset loader
    '''

    def __init__(self, dataset_items, transform=None):
        config = Config()
        self.images_dir = format_data_path(settings.casia_images_dir)
        self.dataset_items = dataset_items
        self.config = config.config
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
        images = self._load_scene(dataset_item)
        return images

    def set_option(self,key, value):
        self.options[key] = value

    def _load_scene(self, dataset_item):
        '''
        Loads scene sequence given a dataset item.
        :param dataset_item: the tuple indicating the dataset item
        :param not_nif_frames: a boolean mask indicating the valid frames. True means valid
        :return: a list of np.arrays containing the scene images
        '''
        p_num, sequence = dataset_item
        scene_folder = os.path.join(self.images_dir,'{:03d}'.format(p_num),sequence)
        def compose_image_filename(angle,i):
            return join(scene_folder , '{}-{:03d}'.format(sequence , angle) , \
                    '{:03d}-{}-{:03d}_frame_{:03d}.jpg'.format(p_num , sequence , angle , i))

        scene_files = []
        if 'valid_indices' in self.options:
            valid_indices = self.options['valid_indices']

            # filter is already applied to the file paths
            for angle in [18 , 54 , 90 , 126 , 162]:
                scene_files += [compose_image_filename(angle , i) for i , valid in enumerate(valid_indices) if valid]
        else:
            for angle in [18 , 54 , 90 , 126 , 162]:
                scene_angle_folder = join(scene_folder , '{}-{:03d}'.format(sequence , angle))
                scene_files += [join(scene_angle_folder , f) for f in listdir(scene_angle_folder) if f.endswith('.jpg')]
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