from os.path import join
from gait_analysis.utils.files import correct_path
# DEFINE YOUR CONFIGURATION HERE
openpose_root = '~/PycharmProjects/openpose_b' # where the bin is located
casia_root = "/home/ron/PycharmProjects/Gait2019/CASIA"
#casia_root = "~/gait_project/CASIAData"
tum_root = "~/gait_project/TUMData"
# configuration = "heatmaps_1"
# configuration = "heatmaps_1"
# configuration = "one_angle"
configuration = "cnn_flows_pretrain"

# ==========================================================
# ==========================================================
# ==========================================================
#   Don't Modify bellow
# ==========================================================
# ==========================================================
# correcting "~"
openpose_root = correct_path(openpose_root) # where the bin is located
casia_root = correct_path(casia_root)
tum_root = correct_path(tum_root)

# TUMGAID FOLDERS
# TODO: REFACTOR TO MAKE TUM AND CASIA SYMMETRIC
###
tumgaid_pose_root = join(tum_root,"preprocessing/pose") # ordered by people folders
tumgaid_flow_root = join(tum_root,"preprocessing/flow") # ordered by people folders
tumgaid_crop_root = join(tum_root,"preprocessing/crop") # ordered by people folders
tumgaid_root = join(tum_root,"images/TUM") # ordered by people folders
tumgaid_annotations_root = join(tum_root,"annotations/TUM") # ordered by people folders
# *********************************************************
# CASIA FOLDERS
casia_pose_dir = join(casia_root,"preprocessing/pose/") # ordered by people folders
casia_heatmap_dir = join(casia_root,"preprocessing/heatmaps/") # ordered by people folders
casia_flow_dir = join(casia_root,"preprocessing/flow/")  # ordered by people folders
casia_crops_dir = join(casia_root,"preprocessing/crops")# ordered by people folders
casia_crops_flow_dir = join(casia_root,"preprocessing/crops_flow")
casia_images_dir = join(casia_root,"images")# ordered by people folders
casia_annotations_dir = join(casia_root,"annotations")# ordered by people folders
# *********************************************************
# calculate_flow = True
# calculate_pose = True
tumgaid_exclude_list = ['back', 'back2']
casia_include_list = ['018', '054','090','126','162']

## LEGACY TUM
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


