import os
import fnmatch


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
