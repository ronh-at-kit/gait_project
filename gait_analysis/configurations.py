default = {
    'pose': {
        'load': True,
        'preprocess': False ,
        'D': 2 ,
        'body_keypoints_include_list': ['LAnkle','RAnkle','LKnee','RKnee','RHip','LHip']
        },
    'flow': {
        'load':True,
        'preprocess' : False,
        'method' : 'dense',
        'load_patches' : True,
        'patch_size' : 5
        },
    'frames':{
        'load':True,
        'preprocess': False,
        'gray_scale' : False,
        'load_tracked' : False,
        'sequences': ['bg','cl','nm'],
        'angles': [90]
    }
}