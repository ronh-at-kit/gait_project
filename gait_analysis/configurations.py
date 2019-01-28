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
        # the complete list is:
        #'body_keypoints_include_list': ['LAnkle' , 'RAnkle' , 'LKnee' , 'RKnee' , 'RHip' , 'LHip' , 'RBigToe' ,
        #                                'LBigToe' , 'RSmallToe' , 'LSmallToe' , 'RHeel' , 'LHeel']
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
        'crops' : False,
        'gray_scale' : False,
        'load_tracked' : False,
        'sequences': ['bg','cl','nm'],
        'angles': [90]
    },
    'heatmaps':{
        'load':False,
        'preprocess': True,
        'body_keypoints_include_list' : ['LAnkle','RAnkle']
    },
    # 'dataset_output' : {
    #     'data': ["scenes","flows","heatmaps"],
    #     'label': "annotations"
    # },
    'transformers':{
        # 'Crop':{'include list':['LAnkle','RAnkle'],'output_size':256,'target':'flows'}
        'SpanImagesList': {'remove':True, 'names': ["heatmaps_LAnkle","heatmaps_RAnkle"],'target': ["heatmaps"]},
        'Rescale': {'output_size' : (640,480), 'target': ["heatmaps_LAnkle","heatmaps_RAnkle"]},
        'AnnotationToLabel': {'target': ["annotations"]},
        'Transpose' : {'swapping': (2, 0, 1) , 'target': ["scenes","flows"]},
        'ToTensor': {'target':["heatmaps_LAnkle","heatmaps_RAnkle","scenes","flows","annotations"]}
    }
}