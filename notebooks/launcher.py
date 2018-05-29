import json
import shutil
from subprocess import call
import datetime
import os
exp_folder = r'/media/sandro/Volume/Datasets/tumgaid/experiments/exp_1/'
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        pass
svm_args = {
    'kernel' : 'linear',
    'C' : 1e-3    
}
tumgaid_args = {
    'load_pose': True,
    'load_pose_options': {
        'D': 2,
        'body_keypoints_include_list': [
            'LAnkle',
            #'RAnkle',
            #'LKnee',
            #'RKnee',
            #'RHip',
            #'LHip',
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
    'max_test': None,
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
    
for t in range(6):
    now = datetime.datetime.now()
    newDirName = now.strftime("%Y_%m_%d-%H%M%S")    
    out_dir = os.path.join(exp_folder, newDirName)
    mkdir_p(out_dir)
    args_dict = {
        'diff_poses' : True,
        'diff_pose_magnitude' : True,
        'temporal_extent' : 1,
        'split_index': 10
        'max_test': None,
        'output-dir' : out_dir,
        'svm_args' : svm_args,
        'tumgaid_args' : tumgaid_args,
    }
    args_dict['temporal_extent'] = t
    with open('args.json', 'w') as f:
        json.dump(args_dict, f)
    run_file(out_dir)