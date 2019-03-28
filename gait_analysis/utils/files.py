import csv
import os
import fnmatch
import glob
# from gait_analysis.Config import Config
import datetime
import logging

def parse_csv(filename, with_header = True):
    '''
    Parse file names and output a dict of lists
    :param filename:
    :return: dict of columns: {'col name': 'values list'}
    '''
    if not os.path.exists(filename):
        logger = logging.getLogger()
        logger.error('File {} doesn\'t exists'.format(filename))
        raise FileExistsError
    with open(filename, 'rt', encoding= 'utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='/')
        columns = {}
        if with_header:
            headers = csvreader.__next__()
            for h in headers:
                columns[h] = []
        else:
            first_row = csvreader.__next__()
            headers = range(0,len(first_row))
            for h,v in  zip(headers,first_row):
                columns[h] = [v]
        for row in csvreader:
            for h, v in zip(headers, row):
                columns[h].append(v)
    return columns

def is_empty_path(path):
    print('=====len(list_all_files(path,\'*\')) = ', len(list_all_files(path,'*')))
    if len(list_all_files(path,'*'))>0:
        return False
    else:
        return True


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
    files_matched = []
    pattern = "*." + extension
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if fnmatch.fnmatch(file, pattern):
                files_matched.append(os.path.join(root,file))
    return files_matched

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

def set_logger(name, config, time_stamp = None, level = 'INFO'):
    if not time_stamp:
        time_stamp = str(datetime.datetime.now()).replace(' ' , '_')
    log_folder = format_data_path(config.config['logger']['log_folder'])
    log_folder = os.path.join(log_folder,'{}-{}'.format(name,time_stamp))
    makedirs(log_folder)

    # selecting the level/parsing level input
    if level == 'INFO':
        level_selected = logging.INFO
    elif level == 'WARN':
        level_selected = logging.WARN
    elif level == 'DEBUG':
        level_selected = logging.DEBUG
    else:
        print('Logging selected critical: from {}'.format(level))
        level_selected = logging.CRITICAL


    logging.basicConfig(
        level=level_selected,
        format="%(asctime)s-[ %(threadName)-12.12s]-[ %(levelname)-5.5s] : %(message)s",
        handlers=[
            logging.FileHandler("{0}/{1}-{2}".format(
                log_folder,time_stamp,
                config.config['logger']['log_file'])),
            logging.StreamHandler()
        ])
    return logging.getLogger(), log_folder