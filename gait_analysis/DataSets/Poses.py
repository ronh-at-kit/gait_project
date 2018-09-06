from os import listdir
from os.path import isfile, join
import gait_analysis.utils.openpose_utils as opu
import numpy as np

from gait_analysis.utils.files import format_data_path
from functools import partial
from torch.utils.data import Dataset, DataLoader



class Poses(Dataset):
    '''
    TumGAID_Dataset loader
    '''

    def __init__(self,
                 dataset_items,
                 poses_path,
                 args_dict,
                 transform=None
                 ):
        '''
        :param dataset_items:
        :param poses_path:
        :param args_dict:
        :param transform torch transform object

        '''
        self.poses_path = format_data_path(poses_path)
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

        # load pose and their valid indices
        pose_keypoints, nif_pos = self._load_pose(dataset_item)

        return pose_keypoints, nif_pos

    def _load_pose(self, dataset_item):

        pose_options = self.options_dict['load_pose_options']
        p_num, sequence = dataset_item

        # format pose folder
        pose_folder = join(self.poses_path,'pose','p{:03d}'.format(p_num), sequence)

        def create_path(i):
            return join(pose_folder, '{:03d}_keypoints.json'.format(i))

        if 'valid_indices' in self.options_dict:
            valid_indices = self.options_dict['valid_indices']
            pose_files = [create_path(i) for i, is_not_nif in enumerate(valid_indices) if valid_indices[i]]
            idx_valid_frames = [i for i, val in enumerate(valid_indices) if val]
        else:
            pose_files = [join(pose_folder, f) for f in listdir(pose_folder) if f.endswith('.json')]
            idx_valid_frames = np.arange(len(pose_files))
            valid_indices = np.asarray([True]*len(pose_files))

        keypoints = map(opu.load_keypoints_from_file, pose_files)
        people = [k['people'] for k in keypoints]

        poses = []

        # there can be frames without poses, which affects the validity of those frames
        # here, we use the validity list from the annotations (nif_pos) and give it to _laod_pose
        # we return the updates list of valid frames and the pose_keypoints
        indices_with_no_poses = []
        for i in range(len(people)):
            # TODO this is very dirty to use try block. You could use if people if to see if its empty
            try:
               poses.append(people[i][0]['pose_keypoints_2d'])
            except:
               indices_with_no_poses.append(i)

        idx_valid_frames = np.asarray(idx_valid_frames)
        indices_with_no_poses = np.asarray(indices_with_no_poses)

        if indices_with_no_poses.any():
            valid_indices[idx_valid_frames[indices_with_no_poses]] = False

        #poses = map(lambda x:  x[0]['pose_keypoints_2d'], people)

        pose_dicts = map(opu.keypoints_to_posedict, poses)

        include_list = pose_options['body_keypoints_include_list']
        func = partial(opu.filter_keypoints, **{'include_list' : include_list,
                                                'return_list' : True,
                                                'return_confidence' : False
                                                })
        poses = map(func, pose_dicts)
        return poses, valid_indices



if __name__ == '__main__':
    pass