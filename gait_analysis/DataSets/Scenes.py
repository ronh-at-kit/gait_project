import os
import glob
from itertools import product
from gait_analysis.utils.data_loading import not_NIF_frame_nums, load_sequence_annotation, remove_nif
import cv2


import numpy as np

from torch.utils.data import Dataset


def format_data_path(data_path):
    '''
    preprocess path containing data
    :param data_path:
    :return: data_path_formated
    '''
    data_path_corrected = data_path

    # expand user home folder if needed
    if data_path.startswith("~"):
        data_path_corrected = os.path.expanduser(data_path)

    # verifies if the folder exists
    if not os.path.exists(data_path_corrected):
        raise ValueError('{} don\'t exist.'.format(data_path_corrected))
    return data_path_corrected

class Scenes(Dataset):
    '''
    TumGAID_Dataset loader
    '''

    def __init__(self, tumgaid_root,
                 tumgaid_annotations_root,
                 args_dict,
                 transform=None
                 ):
        '''
        :param tumgaid_root:
        :param tumgaid_annotations_root:
        :param args_dict:
        :param transform torch transform object

        '''
        tumgaid_root = format_data_path(tumgaid_root) # format_data_path prepares the paths
        tumgaid_annotations_root = format_data_path(tumgaid_annotations_root)

        self.tumgaid_root = tumgaid_root
        self.tumgaid_annotations_root = tumgaid_annotations_root

        annotation_files = sorted(
            glob.glob(
                os.path.join(
                    tumgaid_annotations_root, 'annotation_p*.ods'
                )
            )
        )
        p_nums = map(extract_pnum, annotation_files)
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
        :return:
        '''

        dataset_item = self.dataset_items[idx]
        output = {}

        # returns annotations with NIF
        annotations = self._load_annotation(dataset_item)
        nif_pos = not_NIF_frame_nums(annotations)
        nif_pos = np.array(nif_pos)

        # Loading scene images
        images = self._load_scene(dataset_item, nif_pos)
        output['images'] = images

        # use NIF positions to filter out scene images, flow_maps, and others
        return images


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


    def _load_scene(self, dataset_item, not_nif_frames):
        '''
        Loads scene sequence given a dataset item.
        :param dataset_item: the tuple indicating the dataset item
        :param not_nif_frames: a boolean mask indicating the valid frames. True means valid
        :return: a list of np.arrays containing the scene images
        '''
        p_num, sequence = dataset_item
        sequence_folder = os.path.join(self.tumgaid_root,
                                       'image',
                                       'p{:03d}'.format(p_num),
                                       sequence)
        def create_path(i):
            return os.path.join(sequence_folder, '{:03d}.jpg'.format(i))

        # filter is already applied to the file paths
        scene_files = [create_path(i) for i, is_not_nif in enumerate(not_nif_frames) if is_not_nif]

        def load_imfile(im_file):
            if not os.path.isfile(im_file):
                raise ValueError('{} don\'t exist.'.format(im_file))
            im = cv2.imread(im_file, -1)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            return im
        scene_images = [load_imfile(f) for f in scene_files]
        return scene_images


def extract_pnum(abspath):
    path = os.path.basename(abspath)
    p_num = path[-7:-4]
    return int(p_num)

def construct_image_path(p_num, tumgaid_root):
    image_path = os.path.join(tumgaid_root, 'image', 'p{:03i}'.format(p_num))
    return image_path

def maybe_RGB2GRAY(im_rgb):
    '''
    if the input image has 3 dimensions the image is assumed to be a color image
    and is converted to grayscale.
    Otherwise, the RGB image is returned
    :param im_rgb:
    :return:
    '''
    if len(im_rgb.shape) == 3:
        return cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)
    return im_rgb


if __name__ == '__main__':
    pass