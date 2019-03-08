from cv2 import imread, cvtColor, COLOR_BGR2RGB
import glob
import os
import pandas as pd
import pyexcel as pe
import numpy as np
from gait_analysis.utils.files import list_all_files
from gait_analysis.settings import tumgaid_exclude_list
from gait_analysis.settings import casia_include_list



def list_annotations_files(annotations_dir):
    '''
    Get a list of the path to the annotations files in the  annotations directory
    :param annotations_dir: directory to the annotations directory
    :return annotations_files: list of annotations files
    '''
    annotations_files = sorted(list_all_files(annotations_dir,"ods"))

    return annotations_files
def list_person_folders(images_path, dataset ='TUM'):
    if dataset == 'TUM':
        all_person_folders = sorted(glob.glob(os.path.join(images_path, 'p*')))
    elif dataset == 'CASIA':
        all_person_folders = sorted(glob.glob(os.path.join(images_path, '*/*')))
    return all_person_folders


def list_sequence_folders(person_folder, dataset = 'TUM'):
    '''
    list sequence folders within the given person_folder
    :param person_folder:
    :param dataset: specify the name of the dataset to process.
    :return:
    '''
    if dataset == 'TUM':
        sequence_folders = sorted(glob.glob(os.path.join(person_folder, '*')))
        sequence_folders = [folder for folder in sequence_folders if os.path.basename(folder) not in tumgaid_exclude_list]
    elif dataset == 'CASIA':
        sequence_folders = sorted(glob.glob(os.path.join(person_folder, '*')))
        sequence_folders = [folder for folder in sequence_folders if os.path.basename(folder)[-3:] in casia_include_list]
    else:
        raise ("dataset selected to find sequences folders is not correct. valid values CASIA and TUM. Found: {}".format(dataset) )
    return sequence_folders


def load_sequence_annotation(annotation_file, sequence):
    '''
    Loads the specified sequence from the annotation file.
    Returns it as a pandas dataframe.
    nan values are dropped. this happend mostly at the end of the df
    :param annotation_file:
    :param sequence:
    :return:
    '''
    data = pe.get_dict(file_name=annotation_file, sheets=[sequence])
    df = pd.DataFrame(data)
    df = df.replace('', np.nan).dropna()
    df.frame_id = df.frame_id.astype(int)
    return df

def load_sequence_angle_annotation(annotation_file , sequence , angle):
    '''
    Load the specified sequence angle from the annotaion file
    Returns data as a pandas dataframe
    :param annotation_file:
    :param sequence:
    :param angle:F
    :return: df: sheet dataframe
    '''
    sheet_name = sequence + "-" + '{:03d}'.format(angle)
    data = pe.get_dict(file_name=annotation_file, sheets=[sheet_name])
    df = pd.DataFrame(data)
    df = df.replace('', np.nan).dropna()
    df.frame_id = df.frame_id.astype(int)
    return df

def remove_nif(df, pos):
    '''
    Returns a new data frame only consisting of valid entries. The list 'pos'
    can be used to add additional filters to the list of valid entries.

    Eventually, all values that are NOT_IN_FRAME will be sorted out.
    You can specify a list of True/False values to include / exclude additionl values.
    Items coorresponding to False in the pos list will be sorted out.


    :param df: a dataframe object with 'left_foot' and 'right_foot' values
    :param pos: a list of True and False values
    with the same length as the entries in the data frame
    :return: new data frame with only valid entries
    '''
    df.left_foot.values[~pos] = 'NOT_IN_FRAME'
    df.right_foot.values[~pos] = 'NOT_IN_FRAME'
    df = df[df.left_foot != 'NOT_IN_FRAME']
    new_df = df[df.right_foot != 'NOT_IN_FRAME']
    #print(new_df)
    return new_df

def not_NIF_frame_nums(df):
    '''
    return the frame numbers withOUT NOT_IN_FRAME annotations
    :param df:
    :return: a list of integers
    '''
    return (df.left_foot != 'NOT_IN_FRAME') * (df.right_foot != 'NOT_IN_FRAME')

def as_numeric(df):
    df.left_foot = 1.0 * (df.left_foot == "IN_THE_AIR")
    df.right_foot = 1.0 * (df.right_foot == "IN_THE_AIR")
    return df



def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        pass
        #if exc.errno == errno.EEXIST and os.path.isdir(path):
        #    pass
        #else:
        #    raise

def read_image(im_file):
    if not os.path.isfile(im_file):
        raise ValueError('{} don\'t exist.'.format(im_file))
    im = imread(im_file, -1)
    im = cvtColor(im, COLOR_BGR2RGB)
    return im

def extract_pnum(abspath):
    path = os.path.basename(abspath)
    p_num = path[-7:-4]
    return int(p_num)