import glob
import os
import pandas as pd
import pyexcel as pe
import numpy as np

from gait_analysis.settings import tumgaid_exclude_list


def list_person_folders(TUMGAIDimage_root):
    all_person_folders = sorted(glob.glob(os.path.join(TUMGAIDimage_root, 'image', 'p*')))
    return all_person_folders


def list_sequence_folders(person_folder):
    '''
    list sequence folders within the given person_folder
    :param person_folder:
    :return:
    '''
    sequence_folders = sorted(glob.glob(os.path.join(person_folder, '*')))
    sequence_folders = [folder for folder in sequence_folders if os.path.basename(folder) not in tumgaid_exclude_list]
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


def remove_nif(df):
    df = df[df.left_foot != 'NOT_IN_FRAME']
    new_df = df[df.right_foot != 'NOT_IN_FRAME']
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