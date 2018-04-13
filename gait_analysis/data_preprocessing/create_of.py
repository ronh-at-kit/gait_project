import cv2
import argparse
import os
import glob
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
    # a sequence folder is a folder containing a sequence, like b01 within p001

    person = os.path.basename(person_folder)

    sequence_folders = sorted(glob.glob(os.path.join(person_folder, '*')))
    sequence_folders = [folder for folder in sequence_folders if os.path.basename(folder) not in tumgaid_exclude_list]
    for sequence_folder in sequence_folders:
        sequence = os.path.basename(sequence_folder) # strip path
        output_dir = os.path.join(output_root_dir, person, sequence)
        try_make_dirs(output_dir)
        image_sequence = list_images(sequence_folder, "jpg")
        for i, frame_pair in enumerate(pairwise(image_sequence)):
            prev, next = map(load_image, frame_pair)
            of = calc_of(prev, next)
            write_of(os.path.join(output_dir, 'of_{:03d}.tiff'.format(i)), of)



def create_of_tumgaid(TUMGAIDimage_root, output_dir, only_example=False):
    '''

    :param TUMGAIDimage_root:
    :param output_dir:
    :param only_example: if true, only the first person will be processed to give a preview
    :return:
    '''
    all_person_folders = sorted(glob.glob(os.path.join(TUMGAIDimage_root, 'image', 'p*')))
    if only_example:
        # this is a debug mode
        visit_person_tumgaid(all_person_folders[0], output_dir)
    else:
        for person_folder in all_person_folders:
            visit_person_tumgaid(person_folder, output_dir)


if __name__ == '__main__':
    example_text = '''example:
    
    use --example 1 to only build first person file to give you a preview
    python create_of.py --dataset tum --data-root tumgaid/TUMGAIDimage -o tumgaid/TUMGAIDimage_flow/ --example 1
    
    '''


    parser = argparse.ArgumentParser('preprocessing for datasets', epilog=example_text)
    parser.add_argument('--dataset', type=str, help='which dataset to preprocess')
    parser.add_argument('--data-root', type=str, help='dataset root directory')
    parser.add_argument('-o', type=str, help='outputfolder', dest='output_folder')
    parser.add_argument('--example', type=int, help='will only do a single person to give a preview', default=0)

    args = parser.parse_args()
    args.example = args.example > 0

    if args.dataset == 'tum':
        create_of_tumgaid(args.data_root, args.output_folder, args.example)

