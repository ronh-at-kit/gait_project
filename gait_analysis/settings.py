# DEFINE YOUR CONFIGURATION HERE
openpose_root = '~/gait_project/pose/build/' # where the bin is located

# TUMGAID FOLDERS
# TODO: REFACTOR TO MAKE TUM AND CASIA SYMMETRIC
###
pose_root = '~/gait_project/TUMData/preprocessing/TUM/DatasetA/pose' # where the bin is located
flow_root = '~/gait_project/TUMData/preprocessing/TUM/DatasetA/flow'
raw_root = '~/gait_project/TUMData/preprocessing/TUM/DatasetA/raw'
tumgaid_root = "~/gait_project/TUMData/images/TUM/DatasetA/"
tumgaid_preprocessing_root = "~/gait_project/TUMData/preprocessing/TUM/DatasetA/"
tumgaid_annotations_root = "~/gait_project/TUMData/annotations/TUM/DatasetA/"
# *********************************************************
# CASIA FOLDERS
casia_pose_dir =  "~/gait_project/CASIAData/preprocessing/CASIA/DatasetB/pose"# ordered by people folders
casia_flow_dir =  "~/gait_project/CASIAData/preprocessing/CASIA/DatasetB/flow"# ordered by people folders
casia_raw_dir =  "~/gait_project/CASIAData/preprocessing/CASIA/DatasetB/raw"# ordered by people folders
casia_images_dir = "~/gait_project/CASIAData/images/CASIA/DatasetB/"# ordered by people folders
casia_preprocessing_dir = "~/gait_project/CASIAData//preprocessing/CASIA/DatasetB/"# ordered by people flow pose and raw then by people folders
casia_annotations_dir = "~/gait_project/CASIAData/annotations/CASIA/DatasetB/final_annotations/"# ordered by people folders

# *********************************************************

calculate_flow = False
calculate_pose = True

tumgaid_exclude_list = ['back', 'back2']


# TODO: this is nasty... default should be implemented outside
tumgaid_default_args = {
    'load_pose' : True,
    'load_pose_options' : {
        'D' : 2,
        'body_keypoints_include_list' : ['LAnkle',
                                         'RAnkle',
                                         'LKnee',
                                         'RKnee',
                                         'RHip',
                                         'LHip']
    },
    'load_flow' : True,
    'load_flow_options' :{
        'method' : 'dense',
        'load_patches' : True,
        'load_patch_options' : {
            'patch_size' : 5
        }
    },
    'load_scene' : False,
    'load_scene_options' : {
        'grayscale' : False,
        'load_tracked' : False
    },
    'include_scenes' : ['b01', 'b02', 'n01', 'n02', 's01', 's02'],

}


exp_reports_root = r'/media/sandro/Volume/Datasets/tumgaid/exp_reports/'