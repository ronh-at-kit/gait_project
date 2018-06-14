import os
import glob
from itertools import product
from gait_analysis.utils.data_loading import not_NIF_frame_nums, load_sequence_annotation, remove_nif
import gait_analysis.utils.openpose_utils as opu
import cv2
from functools import partial
from torch.utils.data import Dataset, DataLoader

import numpy as np
from gait_analysis.utils.iterators import pairwise
from gait_analysis.data_preprocessing.preprocess import calc_of


class AbstractGaitDataset:
    # TODO inherit this from pytorch dataset class
    pass


class TumGAID_Dataset(AbstractGaitDataset):
    '''
    TumGAID_Dataset loader
    '''

    def __init__(self, tumgaid_root,
                 tumgaid_preprocessing_root,
                 tumgaid_annotations_root,
                 args_dict,
                 transform=None):

        '''
        :param tumgaid_root:
        :param tumgaid_preprocessing_root:
        :param tumgaid_annotations_root:
        :param args_dict:
        '''
        self.tumgaid_root = tumgaid_root
        self.tumgaid_preprocessing_root = tumgaid_preprocessing_root
        self.tumgaid_annotations_root = tumgaid_annotations_root

        annotation_files = sorted(
            glob.glob(
                os.path.join(
                    tumgaid_annotations_root, 'annotation_p*.ods'
                )
            )
        )
        p_nums = list(map(extract_pnum, annotation_files))
        all_options = list(product(p_nums, args_dict['include_scenes']))

        self.p_nums = p_nums
        self.included_scenes = args_dict['include_scenes']
        self.dataset_items = all_options
        self.options_dict = args_dict
        self.transform = transform

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
        output['pose_keypoints'] = pose_keypoints

        annotations = remove_nif(annotations, nif_pos)

        if self.options_dict['load_flow']:
            flow_maps, scene_images = self._load_flow(dataset_item, nif_pos)
            output['flow_maps'] = flow_maps
        # if self.options_dict['load_pose']:
        #    pose_keypoints = self._load_pose(dataset_item, nif_pos)
        #    output['pose_keypoints'] = pose_keypoints
        if self.options_dict['load_scene']:
            output['scene_images'] = scene_images
            images = self._load_scene(dataset_item, nif_pos)
            output['total_images'] = images

        # if self.transform is not None:
        #            output = self.transform(output)
        # use NIF positions to filter out scene images, flow_maps, and others
        return output, annotations

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

        pose_files = [create_path(i) for i, is_not_nif in enumerate(not_nif_frames) if is_not_nif]
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

        # poses = map(lambda x:  x[0]['pose_keypoints_2d'], people)
        # pose_dicts = []
        # print(type(poses))
        # for i in range(len(poses)):
        #    pose_dicts[i] = opu.keypoints_to_posedict(poses[i])
        pose_dicts = map(opu.keypoints_to_posedict, poses)
        pose_dicts = list(pose_dicts)

        include_list = pose_options['body_keypoints_include_list']
        func = partial(opu.filter_keypoints, **{'include_list': include_list,
                                                'return_list': True,
                                                'return_confidence': False
                                                })

        poses = list(map(func, pose_dicts))

        return poses, not_nif_frames

    def _load_flow(self, dataset_item, not_nif_frames):
        flow_options = self.options_dict['load_flow_options']

        p_num, sequence = dataset_item
        flow_folder = os.path.join(
            self.tumgaid_preprocessing_root,
            'flow',
            'p{:03d}'.format(p_num),
            sequence)

        def create_path(i):
            return os.path.join(flow_folder, 'of_{:03d}.npy'.format(i))

        flow_files = [create_path(i) for i, is_not_nif in enumerate(not_nif_frames) if is_not_nif]
        flow_files = flow_files[:-1] #flow is diff image, therefore there is one less than regular scene frames


        scene_images = self._load_scene(dataset_item, not_nif_frames)
        if flow_options['load_patches']:
            patch_options = flow_options['load_patch_options']
            poses, _ = self._load_pose(dataset_item, not_nif_frames)
            flow_maps = self._load_flow_sequence(flow_files)

            flow_patches = []
            scene_patches = []
            # TODO remove hard-coded slicing of poses because flow_maps always take two image pairs
            # and poses operate on each image
            # otherwise len(flow_maps) and len(pose) would not be the same


            for flow_map, pose in list(zip(flow_maps, poses[:-1])):
                patch = extract_patch_around_points(flow_map, patch_options['patch_size'], pose)
                flow_patches.append(patch)

            for scene_image, pose in list(zip(scene_images, poses[:-1])):
                patch = extract_patch_around_points(scene_image, patch_options['patch_size'], pose)
                scene_patches.append(patch)

            return flow_patches, scene_patches

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

    def _load_flow_sequence(self, flow_files):
        flow = []
        for file in flow_files:
            flow_mat = np.load(file)
            flow.append(flow_mat)
        return flow

    def _calc_flow_sequence(self, image_sequence):
        flow = []
        for prev, next in pairwise(image_sequence):
            prev = maybe_RGB2GRAY(prev)
            next = maybe_RGB2GRAY(next)
            flow.append(calc_of(prev, next))
        return flow



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
            im = cv2.imread(im_file, -1)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            return im

        scene_images = [load_imfile(f) for f in scene_files]
        return scene_images

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

        assert patch.shape[:2] == (2 * patch_size, 2 * patch_size)

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
        return np.zeros((2 * r, 2 * r, i_dim), dtype=np.uint8)
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


def construct_image_path(p_num, tumgaid_root):
    image_path = os.path.join(tumgaid_root, 'image', 'p{:03i}'.format(p_num))
    return image_path


def calc_flow_magnitude(flow_patch, normalize_xy=False):
    '''
    given x and y flow, returns a concatenated array of flow_x, flow_y, flow_magnitude
    :param flow_patch: shape(p, p, 2)
    :return: shape(p, p, 3)
    '''
    # add the magnitude
    of = flow_patch
    ofx, ofy = map(np.squeeze, np.split(of, 2, axis=-1))
    ofmagnitude = np.sqrt(ofx ** 2 + ofy ** 2)
    if normalize_xy:
        ofx /= ofmagnitude
        ofy /= ofmagnitude
    flow_total = np.stack((ofx, ofy, ofmagnitude), axis=2)
    return flow_total


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


class DsetWrapper():
    _dset = None

    def __init__(self, dset):
        self._dset = dset

    def __len__(self):
        return len(self._dset)

    def __getitem__(self, idx):
        data, annotations = self._dset[idx]
        targets = self.annotations_to_targets(annotations)
        return data, targets

    def annotations_to_targets(self, annotations):
        '''
        IN_THE_AIR == TRUE
        '''
        left_foot = annotations.left_foot.values == 'IN_THE_AIR'
        right_foot = annotations.right_foot.values == 'IN_THE_AIR'
        frame_id = annotations.frame_id.values
        return 1.0 * np.stack([frame_id, left_foot, right_foot], axis=1)

class OfPatchesConcat(Dataset):

    def __init__(self, gait_set, concat_default_args ):
        self.gait_set = gait_set
        self.concat_default_args = concat_default_args
        #self.list_seq, self.list_frames = self._set_loading_list()
        #print (self.list_seq)
        #print(self.list_frames)
        self.outputs = self._concatenate_dataset()

    def __len__(self):
        return len(self.outputs['annotations'])

    def __getitem__(self, idx):
        flow = self.outputs['flow'][idx]
        poses = self.outputs['poses'][idx]
        annotation = self.outputs['annotations'][idx]
        return flow,poses,annotation

    def _set_loading_list(self):

        frames_to_concat = self.concat_default_args['frames_to_concat']
        list_seq = []
        list_frames = []

        for seq_idx in range(len(self.gait_set)):
            len_seq = len(self.gait_set[seq_idx])
            items = len_seq - frames_to_concat +1
            list_seq += [seq_idx for x in range(0,items)]
            list_frames += [x for x in range(0,items)]

        return list_seq, list_frames

    def _concatenate_dataset(self):
        output = {}
        flow_concat = []
        pose_concat = []
        annotations_concat = []
        # go through all video sequences
        for seq_idx in range(len(self.gait_set)):
            seq,ant = self.gait_set[seq_idx]

            #go through all the sequences
            middle_idx = np.int((self.concat_default_args['frames_to_concat'] - 1) / 2)
            flow_shape = np.shape(seq['flow_maps'][0][0])
            no_of_poses = len(seq['flow_maps'][0])

            patch_size = self.concat_default_args['load_flow_options']['load_patch_options']['patch_size']
            frames_to_concat = self.concat_default_args['frames_to_concat']

            for i in range(0 , len(seq['flow_maps']) - self.concat_default_args['frames_to_concat'] +1):

                #flow = np.zeros((flow_shape[0],flow_shape[1], flow_shape[2] * no_of_poses * self.concat_default_args['frames_to_concat']))
                #extract the neccessary frames
                flow = seq['flow_maps'][i:i + self.concat_default_args['frames_to_concat']]
                poses = seq['pose_keypoints'][i+1:i + 1 + self.concat_default_args['frames_to_concat']] #+1 bcs OF is frame difference, so OF[0] is differnce between frame[1] and frame[0]

                pose_array = [item for sublist in poses for item in sublist]
                pose_array = np.array(pose_array)

                # flatten list of lists
                flow_array = [item for sublist in flow for item in sublist]
                flow_array = np.array(flow_array)

                flow_array_dims = flow_array.shape
                flow_array = np.reshape(np.transpose(flow_array, (0, 3, 1, 2)), (flow_array_dims[0] * flow_array_dims[3] , 2*patch_size, 2*patch_size)) # 3 bcs of x,y and mag

                #flow_array = np.transpose(flow_array, (1, 2, 0))

                flow_concat.append(flow_array)
                pose_concat.append(pose_array)
                annotations_concat.append(ant[i+middle_idx+1,1:])  #+1 bcs OF is frame difference, so OF[0] is differnce between frame[1] and frame[0]



        output['poses'] = pose_concat
        output['flow'] = flow_concat
        output['annotations'] = annotations_concat
        return output





if __name__ == '__main__':
    pass


