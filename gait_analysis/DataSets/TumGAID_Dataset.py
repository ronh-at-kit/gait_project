import os
import glob
from itertools import product
from gait_analysis.utils.data_loading import not_NIF_frame_nums, load_sequence_annotation, remove_nif
import cv2

import numpy as np
from gait_analysis.utils.iterators import pairwise
from gait_analysis.utils.files import format_data_path
from gait_analysis.utils.data_loading import list_annotations_files
from gait_analysis.data_preprocessing.preprocess_tum import calc_of


from gait_analysis import Scenes, Annotations, PosesTum
from torch.utils.data import Dataset


class TumGAID_Dataset(Dataset):
    '''
    TumGAID_Dataset loader
    '''

    def __init__(self, tumgaid_root,
                 tumgaid_preprocessing_root,
                 tumgaid_annotations_root,
                 args_dict,transform = None):
        '''
        :param tumgaid_root:
        :param tumgaid_preprocessing_root:
        :param tumgaid_annotations_root:
        :param args_dict:
        '''
        self.tumgaid_root = format_data_path(tumgaid_root)
        self.tumgaid_preprocessing_root = format_data_path(tumgaid_preprocessing_root)
        self.tumgaid_annotations_root = format_data_path(tumgaid_annotations_root)
        annotation_files = list_annotations_files(self.tumgaid_annotations_root)
        person_numbers = map(extract_pnum, annotation_files)
        dataset_items = list(product(person_numbers, args_dict['include_scenes']))


        self.person_numbers = person_numbers
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

        output = {}

        # get the annotations
        self.annotations = Annotations(self.dataset_items, self.tumgaid_annotations_root);
        annotations, in_frame_indices = self.annotations[idx]
        # get the poses
        pose_options = self.options_dict
        pose_options['valid_indices'] = in_frame_indices
        poses = PosesTum(self.dataset_items , self.tumgaid_preprocessing_root , pose_options)
        pose_keypoints, valid_indices = poses[idx]
        output['pose_keypoints'] = pose_keypoints

        # clean annotations once again from invalid pose detection
        annotations = remove_nif(annotations, valid_indices)
        if self.options_dict['load_scene'] or self.options_dict['load_flow']:
            scene_options = self.options_dict
            scene_options['valid_indices'] = valid_indices
            scenes = Scenes(self.dataset_items, self.tumgaid_root, scene_options)
            images = scenes[idx]

        if self.options_dict['load_scene']:
            output['images'] = images

        if self.options_dict['load_flow']:
            flow_options = self.options_dict['load_flow_options']
            flow_maps = self._load_flow(images, pose_keypoints, flow_options)
            output['flow_maps'] = flow_maps

        return output, annotations



    def _load_flow(self, images, poses, flow_options):
        if flow_options['load_patches']:
            patch_options = flow_options['load_patch_options']
            flow_maps = self._calc_flow_sequence(images)
            flow_patches = []
            # TODO remove hard-coded slicing of poses because flow_maps always take two image pairs
            # and poses operate on each image
            # otherwise len(flow_maps) and len(pose) would not be the same
            for flow_map, pose in zip(flow_maps, poses[:-1]):
                patch = extract_patch_around_points(flow_map, patch_options['patch_size'], pose)
                flow_patches.append(patch)
            return flow_patches


    def _calc_flow_sequence(self, image_sequence):
        flow = []
        for prev, next in pairwise(image_sequence):
            prev = maybe_RGB2GRAY(prev)
            next = maybe_RGB2GRAY(next)
            flow.append(calc_of(prev, next))
        return flow


def extract_patch_around_points(image_data, patch_size, point_list):
    '''
    :param image_data: np.array representing image data
    :param patch_size: size of patch. square patches are assumed
    :param point_list: a list of pixel coorinates around which to extract the patches
    :return:
    '''
    patches = []

    for arr in point_list:

        x, y = arr.astype(np.uint)
        patch = extract_patch(image_data, patch_size, center_point=(x, y))

        assert patch.shape[:2] == (2*patch_size, 2*patch_size)

        patches.append(patch)
    return patches

def extract_patch(image_data, patch_size, center_point):
    '''
    assuming image_data is [sizex, sizey, channels]
    :param image_data:
    :param patch_size:
    :param center_point:
    :return:
    '''
    cx, cy = center_point
    i_y, i_x, i_dim = image_data.shape

    # Check whetehr the center points are outside the frame
    if (cx > i_x):
        cx = i_x

    if (cy > i_y):
        cy = i_y

    r = patch_size  # for more compact style

    # Check if keypoints are zero, in that case return the zero matrix
    if (cx + cy == 0):
        return np.zeros((2*r,2*r,i_dim), dtype=np.uint8)
    # pad image before patch extraction to also get the correct path size
    # for center_points close to the edge
    else:

        x_start, x_end = np.array([-r, r]) + cx + r
        y_start, y_end = np.array([-r, r]) + cy + r
        image_data = np.pad(image_data, ((r, r), (r, r), (0, 0)), 'constant')

        return image_data[y_start:y_end, x_start:x_end, ...]

def extract_pnum(abspath):
    path = os.path.basename(abspath)
    p_num = path[-7:-4]
    return int(p_num)


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