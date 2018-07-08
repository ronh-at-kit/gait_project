import os
import cv2
from gait_analysis.utils.files import format_data_path
from os import listdir
from os.path import join


from torch.utils.data import Dataset



class Scenes(Dataset):
    '''
    TumGAID_Dataset loader
    '''

    def __init__(self, dataset_items, tumgaid_root, args_dict, transform=None):
        '''
        :param tumgaid_root:
        :param tumgaid_annotations_root:
        :param args_dict:
        :param transform torch transform object

        '''
        self.tumgaid_root = format_data_path(tumgaid_root)
        self.dataset_items = dataset_items
        self.options_dict = args_dict


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
        output['images'] = images

        # use NIF positions to filter out scene images, flow_maps, and others
        return images



    def _load_scene(self, dataset_item):
        '''
        Loads scene sequence given a dataset item.
        :param dataset_item: the tuple indicating the dataset item
        :param not_nif_frames: a boolean mask indicating the valid frames. True means valid
        :return: a list of np.arrays containing the scene images
        '''
        p_num, sequence = dataset_item
        scene_folder = os.path.join(self.tumgaid_root,
                                       'image',
                                       'p{:03d}'.format(p_num),
                                       sequence)
        def create_path(i):
            return os.path.join(scene_folder, '{:03d}.jpg'.format(i))

        if 'valid_indices' in self.options_dict:
            not_nif_frames = self.options_dict['valid_indices']
            # filter is already applied to the file paths
            scene_files = [create_path(i) for i, is_not_nif in enumerate(not_nif_frames) if is_not_nif]
        else:
            scene_files = [join(scene_folder, f) for f in listdir(scene_folder) if f.endswith('.jpg')]

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