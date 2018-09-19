# DEFINE YOUR CONFIGURATION HERE
# openpose_root = '/home/sandro/Projekte/2018_gait/openpose/'
# tumgaid_root = "/media/sandro/Volume/Datasets/tumgaid/TUMGAIDimage/"
# tumgaid_preprocessing_root = "/media/sandro/Volume/Datasets/tumgaid/TUMGAIDimage_preprocessed/"
# tumgaid_annotations_root = "/media/sandro/Volume/Datasets/tumgaid/annotations/"

# TUMGAID FOLDERS
openpose_root = '/home/sandro/Projekte/2018_gait/openpose/'
tumgaid_root = "~/gait_project_folder/TUMData/TUMGAIDimage"
tumgaid_preprocessing_root = "~/gait_project_folder/TUMData/preprocessing"
tumgaid_annotations_root = "~/gait_project_folder/TUMData/annotations"
# *********************************************************
# CASIA FOLDERS
CASIA_IMAGES_DIR = "~/gait_project_folder/casia_images/CASIA/DatasetB"
CASIA_PREPROCESSING_DIR = "~/gait_project_folder/casia_data/preprocessing"
CASIA_ANNOTATIONS_DIR = "~/gait_project_folder/gait_annotations/CASIA/DatasetB/final_annotations"

# *********************************************************

calculate_flow = False
calculate_pose = True

tumgaid_exclude_list = ['back', 'back2']



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