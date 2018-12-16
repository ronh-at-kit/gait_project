import numpy as np
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset
from functools import partial
from gait_analysis.Config import Config
import gait_analysis.utils.openpose_utils as op_utils
import gait_analysis.settings as settings
from gait_analysis.utils.files import format_data_path

class PosesCasia(Dataset):
    '''
    TumGAID_Dataset loader
    '''

    def __init__(self, dataset_items, transform=None):
        # c = Config()
        self.poses_path = format_data_path(settings.casia_pose_dir)
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

        # load pose and their valid indices
        pose_keypoints, nif_pos = self._load_pose(dataset_item)

        return pose_keypoints, nif_pos

    def set_option(self,key, value):
        self.options[key] = value

    def _load_pose(self, dataset_item):

        pose_options = self.config.config['pose']
        grouping = self.config.get_indexing_grouping()
        if grouping == 'person_sequence_angle':
            p_num , sequence , angle = dataset_item
        elif grouping == 'person_sequence':
            # +dev: this is hardcoded need to be define how we want the data.
            # angle = 90
            p_num , sequence = dataset_item
            angle = 90

        # format pose folder
        pose_folder = join(self.poses_path,'{:03d}'.format(p_num), sequence)

        # 1. compose the poses json-filenames in pose_files
        def compose_pose_filename(angle,i):
            # the expected name is:
            # /preprocessing/pose/001/bg-01/bg-01-018/001-bg-01-018_frame_042_keypoints.json
            return join(pose_folder,'{}-{:03d}'.format(sequence,angle),\
                        '{:03d}-{}-{:03d}_frame_{:03d}_keypoints.json'.format(p_num,sequence,angle,i))
        if 'valid_indices' in self.options:
            valid_indices = self.options['valid_indices']
            pose_files = [compose_pose_filename(angle,i) for i, valid in enumerate(valid_indices) if valid]
            idx_valid_frames = [i for i, valid in enumerate(valid_indices) if valid]
        else:
            pose_angle_folder = join(pose_folder,'{}-{:03d}'.format(sequence,angle))
            pose_files = [join(pose_angle_folder, f) for f in listdir(pose_folder) if f.endswith('.json')]
            idx_valid_frames = np.arange(len(pose_files))
            valid_indices = np.asarray([True]*len(pose_files))

        # 2. load the joson files
        keypoints = map(op_utils.load_keypoints_from_file , pose_files)
        people = [k['people'] for k in keypoints]

        # 3. filter out the incomplete pose detection.
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

        pose_dicts = map(op_utils.keypoints_to_posedict , poses)

        include_list = pose_options['body_keypoints_include_list']
        func = partial(op_utils.filter_keypoints , **{'include_list' : include_list,
                                                'return_list' : True,
                                                'return_confidence' : False
                                                      })
        poses = map(func, pose_dicts)
        return poses, valid_indices



if __name__ == '__main__':
    pass