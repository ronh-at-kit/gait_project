import json
import shutil
from subprocess import call
import datetime
import os
from pprint import pprint as prettyprint

import argparse
epilog = '''
    launcher for SVM training to run multiple options at once
'''
parser = argparse.ArgumentParser(epilog=epilog)
parser.add_argument("--exp-folder", default=r'/media/sandro/Volume/Datasets/tumgaid/experiments/toy_exp')

args = parser.parse_args()

exp_folder = args.exp_folder
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        pass
svm_args = {
    'kernel' : 'linear',
    'C' : 1e-3,    
}
tumgaid_args = {
    'load_pose': True,
    'load_pose_options': {
        'D': 2,
        'body_keypoints_include_list': [
            'LAnkle',
            'RAnkle',
            'LKnee',
            'RKnee',
            'RHip',
            'LHip',
        ]
        
    },
    'load_flow': False,
    'load_flow_options': {
        'method': 'dense',
        'load_patches': True,
        'load_patch_options': {
            'patch_size': 5
        }
    },
    'load_scene': False,
    'load_scene_options': {
        'grayscale': False,
        'load_tracked': False
    },
    'include_scenes': ['b01', 'b02', 'n01', 'n02', 's01', 's02'],
}
args_dict = {
    'diff_poses' : True,
    'diff_pose_magnitude' : True,
    'temporal_extent' : 1,
    'split_index': 5,
    'max_test' : None,
    'output-dir' : '.',
    'svm_args' : svm_args,
    'tumgaid_args' : tumgaid_args,
}
def run_file(out_dir):
    cargs = ['jupyter', 'nbconvert',
           '--ExecutePreprocessor.kernel_name=python2',
           '--execute', '--to', 'html',
           '--output-dir', '{}'.format(out_dir),
           'SVM_Training.ipynb']
    call(cargs, cwd='.')
    
    
from itertools import product
body_keypoints_include_list=[
    'LAnkle',
    'RAnkle',
    'LKnee',
    'RKnee',
    'RHip',
    'LHip',
]
all_options = [body_keypoints_include_list[:i] for i in range(1, len(body_keypoints_include_list))]
opt_tuple = list(product(range(6), all_options))
prettyprint(opt_tuple)
for i, opt in enumerate(opt_tuple):
    t, keypoints = opt
    print(t)
    print(keypoints)
    now = datetime.datetime.now()
    newDirName = now.strftime("%Y_%m_%d-%H%M%S")
    newDirName = '{:03d}_{}'.format(i, newDirName)
    out_dir = os.path.join(exp_folder, newDirName)
    mkdir_p(out_dir)
    args_dict = {
        'diff_poses' : True,
        'diff_pose_magnitude' : True,
        'temporal_extent' : t,
        'split_index': 10,
        'max_test': None,
        'output-dir' : out_dir,
        'svm_args' : svm_args,
        'tumgaid_args' : tumgaid_args,
    }
    args_dict['tumgaid_args']['load_pose_options']['body_keypoints_include_list'] = keypoints
    args_dict['temporal_extent'] = t
    with open('args.json', 'w') as f:
        json.dump(args_dict, f, indent=4, sort_keys=True)
    run_file(out_dir)