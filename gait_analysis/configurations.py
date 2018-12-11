default = {
    'pose': {
        'preprocess': True ,
        'D': 2 ,
        'body_keypoints_include_list': ['LAnkle','RAnkle','LKnee','RKnee','RHip','LHip']
        },
    'flow': {
        'preprocess' : True,
        'method' : 'dense',
        'load_patches' : True,
        'patch_size' : 5
        },
    'frames':{
        'preprocess': False,
        'gray_scale' : False,
        'load_tracked' : False,
        'sequences': ['bg','cl','nm'],
        'angles': [90]
    }
}