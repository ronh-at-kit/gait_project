import subprocess
from tqdm import tqdm
import cv2
import os
from utils.files import correct_path
from gait_analysis import settings
from gait_analysis.settings import openpose_root
from gait_analysis.utils.data_loading import list_person_folders, list_sequence_folders
from gait_analysis.utils.iterators import pairwise
from gait_analysis.utils.files import makedirs, list_images
import tifffile
import numpy as np

import settings


def load_image(path, method='cv2'):
    '''
    Load Image function
    :param path: path to the image
    :param method: for code legacy. maybe I don't understand why we need this function
    :return: image
    '''
    if method == 'cv2':
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return im

def write_of(filename, image, method='tiff'):
    '''
    Write optical flow. maybe will have more methods
    :param filename:
    :param image:
    :param method:
    :return:
    '''
    #cv2.imwrite(filename, image)
    if method == 'tiff':
        # add magnitude to third dimension
        image = np.dstack((image, np.sqrt(image[..., 0]**2 + image[..., 1]**2)))
        tifffile.imsave(filename, image)

def calc_of(prevs, next):
    '''
    Calculate of with more with hard-coded parameters.. irrelevant
    :param prevs:
    :param next:
    :return:
    '''
    of = cv2.calcOpticalFlowFarneback(prevs, next, None,
                                      pyr_scale=0.4,
                                      levels=1,
                                      winsize=12,
                                      iterations=2,
                                      poly_n=8,
                                      poly_sigma=1.2,
                                      flags=0)
    return of

def extract_pose_imagedir(image_dir, output_dir):
    '''
    Json format can be seen here
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
    :param image_dir:
    :param output_dir:
    :return:
    '''
    openpose_bin = os.path.join(settings.openpose_root, 'build', 'examples', 'openpose', 'openpose.bin')
    args = [
            openpose_bin,
            "--image_dir", "{}".format(image_dir),
            "--write_json", "{}".format(output_dir),
            "--display", "0",
            "--render_pose", "0"
    ]
    subprocess.call(args, cwd=settings.openpose_root)




def visit_person_sequence_casia(person_folder):
    '''
    Execute preprocessing function person folder.

    :param person_folder: A person folder contains dynamic sequences and predefined angles.
    :return:
    '''

    # a sequence_folder is a folder containing a sequence, like b01 within p001
    person = os.path.basename(os.path.dirname(person_folder))

    sequence_angle_folders = list_sequence_folders(person_folder, dataset='CASIA')

    for sequence_angle_folder in sequence_angle_folders:
        # Extracts the sequence. The sequence is part of the path
        sequence_angle = os.path.basename(sequence_angle_folder)
        sequence = sequence_angle[0:5]
        print('processing sequence {}/{}'.format(person,sequence_angle))

        #extract flow
        if settings.calculate_flow:
            image_sequence = list_images(sequence_angle_folder, "jpg")
            flow_output_dir = os.path.join(settings.casia_flow_dir, person, sequence, sequence_angle)
            makedirs(flow_output_dir)
            for i, frame_pair in enumerate(pairwise(image_sequence)):
                prev, next = map(load_image, frame_pair)
                of = calc_of(prev, next)
                flow_out_filename = 'of_{}_{:03d}.tiff'.format(sequence_angle, i)
                write_of(os.path.join(flow_output_dir, flow_out_filename), of)

        if settings.calculate_pose:
            #extract pody keypoints
            pose_output_dir = os.path.join(settings.casia_pose_dir, person, sequence, sequence_angle)
            makedirs(pose_output_dir)
            # extract_pose_imagedir(sequence_angle_folder, pose_output_dir)


def preprocess_casia(only_example=False):
    '''

    :param image_root: fixed in the settings
    :param output_dir: (fixed in the settings)
    :param only_example: if true, only the first person will be processed to give a preview
    :return:
    '''

    images_dir = settings.casia_images_dir
    person_sequence_folders = list_person_folders(images_dir, dataset='CASIA')

    # if one example is selected: the list of person folders are reduced to 1 sample
    # this is a debug mode.
    if only_example:
        person_sequence_folders = person_sequence_folders[:1]

    for person_folder in tqdm(person_sequence_folders):
        print("processing folder {}".format(person_folder))
        visit_person_sequence_casia(person_folder)


