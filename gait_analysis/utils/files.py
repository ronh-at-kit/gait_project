import os
import fnmatch
import glob

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

def list_all_files(input_path, extension):
    videos = []
    pattern = "*." + extension
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if fnmatch.fnmatch(file, pattern):
                videos.append(os.path.join(root,file))
    # print(videos)
    return videos

def correct_path(path):
    '''
    Correct the path when it statrs the with ~ (LINUX MAXOS case)
    :param path:
    :return:
    '''
    if path.startswith('~'):
        path = os.path.expanduser(path)
    return os.path.abspath(path)


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('folder exists already. Please double check and maybe delete folder {}'.format(path))
        pass

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
