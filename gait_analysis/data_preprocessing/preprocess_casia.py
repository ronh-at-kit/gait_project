import subprocess
import warnings
import cv2
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

import gait_analysis.settings as settings
from gait_analysis.utils.data_loading import list_person_folders, list_sequence_folders
from gait_analysis.utils.iterators import pairwise
from gait_analysis.utils.files import makedirs, list_images
from gait_analysis import fileUtils
from gait_analysis.utils.ui import query_yes_no
from gait_analysis.Config import Config
c = Config()
CONFIG  = c.config


def abort_overwrite():
    '''
    Warns over the data content in file that may be overwrite. PREVENT TO DESTROY INFORMATION.
    :return:
    '''
    # extract flow
    if CONFIG['flow']['preprocess'] and not fileUtils.is_empty_path(settings.casia_flow_dir):
        abort = query_yes_no('The path {} contains files. The process will overwrite data. Do you want to abort?'.format(settings.casia_flow_dir))
        if abort:
            return True
    if CONFIG['pose']['preprocess'] and not fileUtils.is_empty_path(settings.casia_pose_dir):
        abort = query_yes_no('The path {} contains files. The process will overwrite data. Do you want to abort?'.format(settings.casia_pose_dir))
        if abort:
            return True
    # if no for all return false = total false
    return False

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

def write_of(filename, flow, method='tiff'):
    '''
    Write optical flow. maybe will have more methods
    :param filename:
    :param image:
    :param method:
    :return:
    '''

    norm = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2) #calculate norm as the third parameter

    ''' 
    These are max and min values which have been found experimentally on a test dataset. The test dataset 
    comprised persons 50 - 59. Mix and min values have been evaluated for all 100 sequences.
    The values below are approximately the 0.9 percentiles of the max or min values over all 100 sequences. Numbers 
    have been rounded and have been made symmetrical.    
    '''
    hor_max = 24
    hor_min = -24
    ver_max = 66
    ver_min = -66
    mag_max = 74

    # Clip the values so that they don't exceed the min or max values.
    horz =  np.clip(flow[:,:,0], hor_min, hor_max)
    vert =  np.clip(flow[:,:,1], ver_min, ver_max)
    magn =  np.clip(norm, 0, mag_max)

    # Normalize them to the range 0...255
    horz = np.around(255*((horz - hor_min) / (hor_max - hor_min)))
    vert = np.around(255 * ((vert - ver_min) / (ver_max - ver_min)))
    magn = np.around(255 * ((magn) / (mag_max)))

    # Save image with PIL in order to make use of additional compression. The achieved compression factor is around
    # factor 50, compared to the regular tiff-format (the raw data).
    flow_shape = np.shape(horz)
    img = np.zeros((flow_shape[0], flow_shape[1], 3))
    img[:, :, 0] = horz
    img[:, :, 1] = vert
    img[:, :, 2] = magn

    img_pil = Image.fromarray(np.uint8(img))
    img_pil.save(filename, quality=100, optimize=True)

    # legacy
    # if method == 'tiff':
    #    # add magnitude to third dimension
    #    flow = np.dstack((flow, np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
    #    tifffile.imsave(filename, flow)


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

def extract_pose_imagedir(image_dir, pose_dir = None, headmaps_dir = None):
    '''
    Json format can be seen here
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
    example:
    ./build/examples/openpose/openpose.bin --image_dir /home/ron/Dokumente/Datasets/Gait/CASIA/images/006/bg-01/bg-01-090/ \
                --write_json ~/Dokumente/Datasets/Gait/CASIA/preprocessing/pose/006/bg-01/bg-01-090/  \
                --render_pose 0 --display 0 --heatmaps_add_parts true \
                --write_heatmaps ~/Dokumente/Datasets/Gait/CASIA/preprocessing/heatmaps/006/bg-01/bg-01-090/
    :param image_dir:
    :param output_dir:
    :return:
    '''
    if pose_dir and headmaps_dir:
        openpose_bin = os.path.join(settings.openpose_root, 'build', 'examples', 'openpose', 'openpose.bin')
        args = [
                openpose_bin,
                "--image_dir", "{}".format(image_dir),
                "--write_json", "{}".format(pose_dir),
                "--heatmaps_add_parts", "true",
                "--write_headmap" , "{}".format(headmaps_dir) ,
                "--display", "0",
                "--render_pose", "0"
        ]
    elif pose_dir:
        openpose_bin = os.path.join(settings.openpose_root , 'build' , 'examples' , 'openpose' , 'openpose.bin')
        args = [
            openpose_bin ,
            "--image_dir" , "{}".format(image_dir) ,
            "--write_json" , "{}".format(pose_dir) ,
            "--display" , "0" ,
            "--render_pose" , "0"
        ]
    elif headmaps_dir:
        openpose_bin = os.path.join(settings.openpose_root , 'build' , 'examples' , 'openpose' , 'openpose.bin')
        args = [
            openpose_bin ,
            "--image_dir" , "{}".format(image_dir) ,
            "--write_headmap" , "{}".format(headmaps_dir) ,
            "--display" , "0" ,
            "--render_pose" , "0"
        ]
    else:
        return None
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
        if CONFIG['flow']['preprocess']:
            image_sequence = list_images(sequence_angle_folder, "jpg")
            flow_output_dir = os.path.join(settings.casia_flow_dir, person, sequence, sequence_angle)
            makedirs(flow_output_dir)
            for i, frame_pair in enumerate(pairwise(image_sequence)):
                prev, next = map(load_image, frame_pair)
                of = calc_of(prev, next)
                # flow_out_filename = 'of_{}_{:03d}.tiff'.format(sequence_angle, i)
                flow_out_filename = '{}-{}_frame_{:03d}_flow.png'.format(person,sequence_angle, i)
                write_of(os.path.join(flow_output_dir, flow_out_filename), of)

        if CONFIG['pose']['preprocess'] and CONFIG['heatmaps']['preprocess']:
            #extract pody keypoints
            pose_output_dir = os.path.join(settings.casia_pose_dir, person, sequence, sequence_angle)
            heatmaps_output_dir = os.path.join(settings.casia_heatmaps_dir , person , sequence , sequence_angle)
            makedirs(pose_output_dir)
            extract_pose_imagedir(sequence_angle_folder, pose_dir=pose_output_dir, heatmaps_dir = heatmaps_output_dir)
        elif CONFIG['pose']['preprocess']:
            # extract pody keypoints
            pose_output_dir = os.path.join(settings.casia_pose_dir , person , sequence , sequence_angle)
            makedirs(pose_output_dir)
            extract_pose_imagedir(sequence_angle_folder , pose_dir=pose_output_dir)
        elif CONFIG['heatmaps']['preprocess']:
            # extract pody keypoints
            heatmaps_output_dir = os.path.join(settings.casia_heatmaps_dir , person , sequence , sequence_angle)
            makedirs(pose_output_dir)
            extract_pose_imagedir(sequence_angle_folder , heatmaps_dir=heatmaps_output_dir)


def preprocess_casia(only_example=False):
    '''

    :param image_root: fixed in the settings
    :param output_dir: (fixed in the settings)
    :param only_example: if true, only the first person will be processed to give a preview
    :return:
    '''

    images_dir = settings.casia_images_dir
    print("=======>>>>  images_dir = {}".format(images_dir))
    print("I am in the general version")
    person_sequence_folders = list_person_folders(images_dir, dataset='CASIA')

    # Waring to don't overwrite:
    if abort_overwrite():
        # if user
        warnings.warn("pre-nprocess aborted by user! ", UserWarning)
        return
    # if one example is selected: the list of person folders are reduced to 1 sample
    # this is a debug mode.
    if only_example:
        person_sequence_folders = person_sequence_folders[0:10]
    for person_folder in tqdm(person_sequence_folders):
        print("processing folder {}".format(person_folder))
        visit_person_sequence_casia(person_folder)

