import os
import glob
from itertools import product
from gait_analysis.utils.data_loading import not_NIF_frame_nums, load_sequence_annotation, remove_nif
import gait_analysis.utils.openpose_utils as opu
import cv2
from functools import partial

import numpy as np
from gait_analysis.utils.iterators import pairwise
from gait_analysis.data_preprocessing.preprocess import calc_of

from torch.utils.data import Dataset, DataLoader


class Poses(Dataset):
    '''
    TumGAID_Dataset loader
    '''

    def __init__(self,
                 tumgaid_preprocessing_root,
                 tumgaid_annotations_root,
                 args_dict,
                 transform=None
                 ):
        '''
        :param tumgaid_root:
        :param tumgaid_preprocessing_root:
        :param tumgaid_annotations_root:
        :param args_dict:
        :param transform torch transform object

        '''
        tumgaid_preprocessing_root = format_data_path(tumgaid_preprocessing_root)
        tumgaid_annotations_root = format_data_path(tumgaid_annotations_root)

        self.tumgaid_preprocessing_root = tumgaid_preprocessing_root
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

        # there can be frames without poses, which affects the validity of those frames
        # here, we use the validity list from the annotations (nif_pos) and give it to _laod_pose
        # we return the updates list of valid frames and the pose_keypoints
        pose_keypoints, nif_pos = self._load_pose(dataset_item, nif_pos)
        pose_keypoints

        # use NIF positions to filter out scene images, flow_maps, and others
        return pose_keypoints

    def _load_pose(self, dataset_item, not_nif_frames):
        pose_options = self.options_dict['load_pose_options']
        p_num, sequence = dataset_item
        pose_folder = os.path.join(
            self.tumgaid_preprocessing_root,
            'pose',
            'p{:03d}'.format(p_num),
            sequence)

        def create_path(i):
            return os.path.join(pose_folder, '{:03d}_keypoints.json'.format(i))
        pose_files= [create_path(i) for i, is_not_nif in enumerate(not_nif_frames) if is_not_nif]
        idx_valid_frames = [i for i, val in enumerate(not_nif_frames) if val]

        keypoints = map(opu.load_keypoints_from_file, pose_files)
        people = [k['people'] for k in keypoints]

        poses = []
        idx_no_poses = []
        for i in range(len(people)):
            # TODO this is very dirty to use try block. You could use if people if to see if its empty
           try:
               poses.append(people[i][0]['pose_keypoints_2d'])
           except:
               idx_no_poses.append(i)

        idx_valid_frames = np.asarray(idx_valid_frames)
        idx_no_poses = np.asarray(idx_no_poses)

        if idx_no_poses.any():

            not_nif_frames[idx_valid_frames[idx_no_poses]] = False

        #poses = map(lambda x:  x[0]['pose_keypoints_2d'], people)

        pose_dicts = map(opu.keypoints_to_posedict, poses)

        include_list = pose_options['body_keypoints_include_list']
        func = partial(opu.filter_keypoints, **{'include_list' : include_list,
                                                'return_list' : True,
                                                'return_confidence' : False
                                                })
        poses = map(func, pose_dicts)
        return poses, not_nif_frames

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

def extract_pnum(abspath):
    path = os.path.basename(abspath)
    p_num = path[-7:-4]
    return int(p_num)

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


if __name__ == '__main__':
    pass