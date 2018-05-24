openpose_root = '/home/ron/PycharmProjects/openpose'

tumgaid_root = "/home/ron/Dokumente/Datasets/Gait/TumGaid/TUMGAIDimage/"
tumgaid_preprocessing_root = "/home/ron/Dokumente/Datasets/Gait/TumGaid/TUMGAIDimage/preprocessing/"
tumgaid_annotations_root = "/home/ron/Dokumente/Datasets/Gait/TumGaid/annotations/"
tumgaid_annotations_root_train = "/home/ron/Dokumente/Datasets/Gait/TumGaid/annotations/train/"
tumgaid_annotations_root_test = "/home/ron/Dokumente/Datasets/Gait/TumGaid/annotations/test/"

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
    'load_flow' : False,
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