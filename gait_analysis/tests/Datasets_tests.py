from utils.Datasets import TumGAID_Dataset


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


def test_flow_maps():
    import settings as S
    tumgaid_default_args = {
        'load_pose': True,
        'load_pose_options': {
            'D': 2,
            'body_keypoints_include_list': ['LAnkle',
                                            'RAnkle',
                                            'LKnee',
                                            'RKnee',
                                            'RHip',
                                            'LHip']
        },
        'load_flow': True,
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
    tg_dset = TumGAID_Dataset(S.tumgaid_root,
                              S.tumgaid_preprocessing_root,
                              S.tumgaid_annotations_root,
                              tumgaid_default_args)
    output, annotations = tg_dset[0]
    for k in output['flow_maps']:
        for kk in k:
            assert (kk.shape == (10, 10, 2)), '{}'.format(kk.shape)


# TODO add CI