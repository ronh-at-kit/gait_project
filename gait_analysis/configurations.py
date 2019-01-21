default = {
    'indexing':{
        #'grouping': 'person_sequence_angle',
        'selection': 'manual_people_sequence',     #  => 'auto'= by final annotation or
                                 #  => 'manual_people' = uses 'people' list
                                 #  => 'manual_people_sequence' uses combination of two lists 'people' and 'sequences'
        'people_selection': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'sequences_selection': ['bg-01', 'cl-01']
        },
    'pose': {
        'load': False,
        'preprocess': True ,
        'D': 2 ,
        'body_keypoints_include_list': ['LAnkle','RAnkle','LKnee','RKnee','RHip','LHip']
        },
    'flow': {
        'load':False,
        'preprocess' : True,
        'method' : 'dense',
        'load_patches' : True,
        'patch_size' : 5
        },
    'scenes':{
        'load':True,
        'preprocess': False,
        'gray_scale' : False,
        'load_tracked' : False,
        'sequences': ['bg','cl','nm'],
        'angles': [90]
    },
    'heatmaps':{
        'load':False,
        'preprocess': True
    },
    'dataloaders':{
        'Rescale': {'output_size' : 256, 'target': 'heatmaps'},
        'ToTensor': {}
    }
}