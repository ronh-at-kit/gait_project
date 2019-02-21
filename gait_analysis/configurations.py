default = {
    'indexing':{
        #'grouping': 'person_sequence_angle',
        'selection': 'manual_people_sequence',     #  => 'auto'= by final annotation or
                                 #  => 'manual_people' = uses 'people' list
                                 #  => 'manual_people_sequence' uses combination of two lists 'people' and 'sequences'
        'people_selection': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'sequences_selection': ['bg-01']
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
        # 'SpanImagesList': {'remove':True, 'names': ["heatmaps_LAnkle","heatmaps_RAnkle"],'target': ["heatmaps"]},
        #'Rescale': {'output_size' : (640,480), 'target': ["heatmaps_LAnkle","heatmaps_RAnkle"]},
        'AnnotationToLabel': {'target': ["annotations"]},
        # 'Transpose': {'swapping': (2, 0, 1), 'target': ["scenes", "flows"]},
        'Transpose' : {'swapping': (2, 0, 1) , 'target': ["scenes"]},
        #'DimensionResize' : {'dimension': 10, 'target': ["heatmaps_LAnkle","heatmaps_RAnkle","scenes","flows","annotations"]},
        'DimensionResize' : {'dimension': 10, 'target': ["scenes","annotations"]},
        'ToTensor': {'target':["scenes","annotations"]}
    }
}

scenes = {
    'indexing':{
        #'grouping': 'person_sequence_angle',
        'selection': 'manual_people_sequence',     #  => 'auto'= by final annotation or
                                 #  => 'manual_people' = uses 'people' list
                                 #  => 'manual_people_sequence' uses combination of two lists 'people' and 'sequences'
        'people_selection': [1],
        #'sequences_selection': ['nm-01']
        'sequences_selection': ['bg-01','bg-02','cl-01','cl-02','nm-01','nm-02','nm-03','nm-04','nm-05','nm-06']
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
        # 'SpanImagesList': {'remove':True, 'names': ["heatmaps_LAnkle","heatmaps_RAnkle"],'target': ["heatmaps"]},
        #'Rescale': {'output_size' : (640,480), 'target': ["heatmaps_LAnkle","heatmaps_RAnkle"]},
        'AnnotationToLabel': {'target': ["annotations"]},
        # 'Transpose': {'swapping': (2, 0, 1), 'target': ["scenes", "flows"]},
        'Transpose' : {'swapping': (2, 0, 1) , 'target': ["scenes"]},
        #'DimensionResize' : {'dimension': 10, 'target': ["heatmaps_LAnkle","heatmaps_RAnkle","scenes","flows","annotations"]},
        'DimensionResize' : {'dimension': 10, 'target': ["scenes","annotations"]},
        'ToTensor': {'target':["scenes","annotations"]}
    }
}


flows = {
    'indexing':{
        #'grouping': 'person_sequence_angle',
        'selection': 'manual_people_sequence',     #  => 'auto'= by final annotation or
                                 #  => 'manual_people' = uses 'people' list
                                 #  => 'manual_people_sequence' uses combination of two lists 'people' and 'sequences'
        'people_selection': [1],
        #'sequences_selection': ['nm-01']
        'sequences_selection': ['bg-01','bg-02','cl-01','cl-02','nm-01','nm-02','nm-03','nm-04','nm-05','nm-06']
        },
    'pose': {
        'load': False,
        'preprocess': False,
        'D': 2 ,
        # the complete list is:
        #'body_keypoints_include_list': ['LAnkle' , 'RAnkle' , 'LKnee' , 'RKnee' , 'RHip' , 'LHip' , 'RBigToe' ,
        #                                'LBigToe' , 'RSmallToe' , 'LSmallToe' , 'RHeel' , 'LHeel']
        'body_keypoints_include_list': ['LAnkle','RAnkle','LKnee','RKnee','RHip','LHip']
        },
    'flow': {
        'load':True,
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
        'sequences': ['nm'],
        'angles': [90]
    },
    'heatmaps':{
        'load':False,
        'preprocess': False,
        'body_keypoints_include_list' : ['LAnkle','RAnkle']
    },
    # 'dataset_output' : {
    #     'data': ["flows"],
    #     'label': "annotations"
    # },
    'transformers':{
        # 'Crop':{'include list':['LAnkle','RAnkle'],'output_size':256,'target':'flows'}
        #'SpanImagesList': {'remove':False, 'names': ["heatmaps_LAnkle","heatmaps_RAnkle"],'target': ["heatmaps"]},
        'Rescale': {'output_size' : (480,640), 'target': ["flows"]},
        'AnnotationToLabel': {'target': ["annotations"]},
        'Transpose' : {'swapping': (2, 1, 0) , 'target': ["flows"]},
        'DimensionResize' : {'dimension': 10, 'target': ["flows","annotations"]},
        'ToTensor': {'target':["flows","annotations"]}
    }
}