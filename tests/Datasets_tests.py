from gait_analysis.Datasets import TumGAID_Dataset
from gait_analysis import settings as S



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
            'patch_size' : 10
        }
    },
    'load_scene' : False,
    'load_scene_options' : {
        'grayscale' : False,
        'load_tracked' : False
    },
    'include_scenes' : ['b01', 'b02', 'n01', 'n02', 's01', 's02'],

}


def test_tumgaid():
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

    assert len(output['pose_keypoints']) == len(annotations)
    # flow maps are always pairwise, therfore -1
    assert len(output['flow_maps']) == (len(annotations) - 1)


# TODO test case that pose it outside image


# TODO add CI


if __name__ == '__main__':
    test_tumgaid()