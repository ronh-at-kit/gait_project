import subprocess
from tqdm import tqdm
import cv2
import argparse
import os
import glob

from data_preprocessing import settings
from settings import openpose_root
from utils.iterators import pairwise
import tifffile
import numpy as np


# folders within each person folder that should be excluded from optical flow calculation
# for example back, back2
tumgaid_exclude_list = ['back', 'back2']


# class OpticalFlowCalculator:
#     def __init__(self):
#         pass
#
#     def calc_of(self, prevs, next):
#         '''
#         returns optical flow
#         :param prevs:
#         :param next:
#         :return:
#         '''
#         pass
#
#
# class OF_Farneback(OpticalFlowCalculator):
#     of_args = {}
#     def __init__(self):
#         super(OF_Farneback, self).__init__()
#
#     def calc_of(self, prevs, next):
#         of = cv2.calcOpticalFlowFarneback(prevs, next, **self.of_args)
#         return of


def load_image(path, method='cv2'):
    if method == 'cv2':
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return im

def write_of(filename, image, method='tiff'):
    #cv2.imwrite(filename, image)
    if method == 'tiff':
        # add magnitude to third dimension
        image = np.dstack((image, np.sqrt(image[..., 0]**2 + image[..., 1]**2)))
        tifffile.imsave(filename, image)

def calc_of(prevs, next):
    of = cv2.calcOpticalFlowFarneback(prevs, next, None,
                                      pyr_scale=0.4,
                                      levels=1,
                                      winsize=12,
                                      iterations=2,
                                      poly_n=8,
                                      poly_sigma=1.2,
                                      flags=0)
    return of


def list_images(dir, extension, sort=True):
    '''
    return a list of all images with the given extension within the specified folder
    :param dir:
    :param extension: extension without dot, like 'png'
    :param sorted: if the list should be sorted
    :return:
    '''
    image_list = glob.glob(os.path.join(dir, '*.{}'.format(extension)))
    if sort:
        return sorted(image_list)
    return image_list

def try_make_dirs(path):
    try:
        os.makedirs(path)
    except:
        print('folder exists already. Please double check and maybe delete folder {}'.format(path))
        pass



def visit_person_tumgaid(person_folder, output_root_dir):

    # a sequence_folder is a folder containing a sequence, like b01 within p001
    person = os.path.basename(person_folder)

    sequence_folders = sorted(glob.glob(os.path.join(person_folder, '*')))
    sequence_folders = [folder for folder in sequence_folders if os.path.basename(folder) not in tumgaid_exclude_list]
    for sequence_folder in sequence_folders:
        sequence = os.path.basename(sequence_folder) # strip path
        print('processing sequence {}'.format(sequence))
        flow_output_dir = os.path.join(output_root_dir, 'flow', person, sequence)
        image_sequence = list_images(sequence_folder, "jpg")

        #extract flow
        if settings.calculate_flow:
            try_make_dirs(flow_output_dir)
            for i, frame_pair in enumerate(pairwise(image_sequence)):
                prev, next = map(load_image, frame_pair)
                of = calc_of(prev, next)
                write_of(os.path.join(flow_output_dir, 'of_{:03d}.tiff'.format(i)), of)

        if settings.calculate_pose:
            #extract pody keypoints
            pose_output_dir = os.path.join(output_root_dir, 'pose', person, sequence)
            try_make_dirs(pose_output_dir)
            extract_pose_imagedir(sequence_folder, pose_output_dir)



def preprocess_tumgaid(TUMGAIDimage_root, output_dir, only_example=False):
    '''

    :param TUMGAIDimage_root:
    :param output_dir:
    :param only_example: if true, only the first person will be processed to give a preview
    :return:
    '''
    all_person_folders = sorted(glob.glob(os.path.join(TUMGAIDimage_root, 'image', 'p*')))
    if only_example:
        all_person_folders = all_person_folders[:1]
        # this is a debug mode
        # visit_person_tumgaid(all_person_folders[0], output_dir):
    for person_folder in tqdm(all_person_folders):
        print("processing folder {}".format(person_folder))
        visit_person_tumgaid(person_folder, output_dir)




def extract_pose_imagedir(image_dir, output_dir):
    openpose_bin = os.path.join(openpose_root, 'build', 'examples', 'openpose', 'openpose.bin')
    args = [
            openpose_bin,
            "--image_dir", "{}".format(image_dir),
            "--write_json", "{}".format(output_dir),
            "--display", "0",
            "--render_pose", "0"
    ]
    subprocess.call(args, cwd=openpose_root)