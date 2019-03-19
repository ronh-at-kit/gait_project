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
        'preprocess': False ,
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
        'preprocess': False,
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
        'grouping': 'person_sequence_angle',
        'selection': 'manual_people',     #  => 'auto'= by final annotation or
                                 #  => 'manual_people' = uses 'people' list
                                 #  => 'manual_people_sequence' uses combination of two lists 'people' and 'sequences'
        'people_selection': [1,2,3,5,6,7,8,9,10,40,41,42],
        #'sequences_selection': ['bg-01']
        #'sequences_selection': ['bg-01','bg-02','cl-01','cl-02','nm-01','nm-02','nm-03','nm-04','nm-05','nm-06']
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
    'dataset_output' : {
        'data': ["scenes"],
        'label': "annotations"
    },
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
    },
    'network': {
        'learning_rate': 0.01,
        'validation_split': 0.2,
        'momentum': 0.9,
        'randomized_seed': 10,
        'shuffle_dataset': False,
        'epochs': 100,
        'NR_LSTM_UNITS': 2 ,
        'IMAGE_INPUT_SIZE_W': 640 ,
        'IMAGE_INPUT_SIZE_H': 480 ,
        'IMAGE_AFTER_CONV_SIZE_W': 18 ,
        'IMAGE_AFTER_CONV_SIZE_H': 13 ,
        'LSTM_IO_SIZE': 18 * 13,
        'LSTM_HIDDEN_SIZE': 18 * 13,
        'RGB_CHANNELS': 3,
        'TIMESTEPS': 10,  # size videos
        'BATCH_SIZE': 1, # until now just batch_size = 1
        'device': "cuda:0"
    },
    'logger':{
        'log_file': 'scenes_20_people_attemp_1.log',
        'log_folder': '~/gait_project/logs',
        'plot_file': 'scenes_scenes_20_people_attemp_1.png'
    }
}

scenes_40 = {
    'indexing':{
        'grouping': 'person_sequence_angle',
        'selection': 'manual_people',     #  => 'auto'= by final annotation or
                                 #  => 'manual_people' = uses 'people' list
                                 #  => 'manual_people_sequence' uses combination of two lists 'people' and 'sequences'
        'people_selection': [1,2,3,4,5,6,7,8,9,10,41,42,42,43,44,45,46,47,48,49,50],
        'sequences_selection': ['bg-01']
        #'sequences_selection': ['bg-01','bg-02','cl-01','cl-02','nm-01','nm-02','nm-03','nm-04','nm-05','nm-06']
        },
    'annotations':{
        'preprocess': True
    },
    'pose': {
        'load': False,
        'preprocess': False ,
        'D': 2 ,
        # the complete list is:
        #'body_keypoints_include_list': ['LAnkle' , 'RAnkle' , 'LKnee' , 'RKnee' , 'RHip' , 'LHip' , 'RBigToe' ,
        #                                'LBigToe' , 'RSmallToe' , 'LSmallToe' , 'RHeel' , 'LHeel']
        'body_keypoints_include_list': ['LAnkle','RAnkle','LKnee','RKnee','RHip','LHip']
        },
    'flow': {
        'load':False,
        'preprocess' : False,
        'method' : 'dense',
        'load_patches' : True,
        'patch_size' : 5,
        },
    'scenes':{
        'load': True,
        'preprocess': False,
        'crops' : False,
        'gray_scale' : False,
        'load_tracked' : False,
        'sequences': ['bg','cl','nm'],
        'angles': [90]
    },
    'heatmaps':{
        'load':False,
        'preprocess': False,
        'body_keypoints_include_list' : ['LAnkle','RAnkle']
    },
    'dataset_output' : {
        'data': ["scenes"],
        'label': "annotations"
    },
    'transformers':{
        # 'Crop':{'include list':['LAnkle','RAnkle'],'output_size':256,'target':'flows'}
        # 'SpanImagesList': {'remove':True, 'names': ["heatmaps_LAnkle","heatmaps_RAnkle"],'target': ["heatmaps"]},
        #'Rescale': {'output_size' : (640,480), 'target': ["heatmaps_LAnkle","heatmaps_RAnkle"]},
        'AnnotationToLabel': {'target': ["annotations"]},
        # 'Transpose': {'swapping': (2, 0, 1), 'target': ["scenes", "flows"]},
        'Transpose' : {'swapping': (2, 0, 1) , 'target': ["scenes"]},
        #'DimensionResize' : {'dimension': 10, 'target': ["heatmaps_LAnkle","heatmaps_RAnkle","scenes","flows","annotations"]},
        'DimensionResize' : {'dimension': 40, 'target': ["scenes","annotations"]},
        'ToTensor': {'target':["scenes","annotations"]}
    },
    'network': {
        'learning_rate': 0.007,
        'validation_split': 0.2,
        'momentum': 0.9,
        'randomized_seed': 10,
        'shuffle_dataset': False,
        'epochs': 2,
        'NR_LSTM_UNITS': 2 ,
        'IMAGE_INPUT_SIZE_W': 640 ,
        'IMAGE_INPUT_SIZE_H': 480 ,
        'IMAGE_AFTER_CONV_SIZE_W': 18 ,
        'IMAGE_AFTER_CONV_SIZE_H': 13 ,
        'LSTM_IO_SIZE': 18 * 13,
        'LSTM_HIDDEN_SIZE': 18 * 13,
        'RGB_CHANNELS': 3,
        'TIMESTEPS': 40,  # size videos
        'BATCH_SIZE': 1, # batch_size
        'device': "cuda:1",
        'many_to_fewer': 10 # this used for many to fewer model. EXAMPLE =20, means from the 20th to the end
    },
    'logger':{
        'log_file': 'scenes_20_people_20_timesteps_attemp_1.log',
        'log_folder': '~/gait_project/logs',
        'plot_file': 'scenes_20_people_20_timesteps_attemp_1.png',
        'model_file': 'scenes_20_people_20_timesteps_attemp_1.tar'
    }
}

one_angle = {
    'indexing':{
        'grouping': 'person_sequence',
        'selection': 'manual_people_sequence',     #  => 'auto'= by final annotation or
                                 #  => 'manual_people' = uses 'people' list
                                 #  => 'manual_people_sequence' uses combination of two lists 'people' and 'sequences'
        'people_selection': [1,2], #[1, 2, 3, 5, 6, 7, 8, 9, 10],
        'sequences_selection': ['bg-01','cl-01','nm-01'], # ['bg-01','bg-02','cl-01','cl-02','nm-01','nm-02']
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
    'dataset_output' : {
        'data': ["scenes"],
        'label': "annotations"
    },
    'transformers':{
        'AnnotationToLabel': {'target': ["annotations"]},
        'Transpose' : {'swapping': (2, 0, 1) , 'target': ["scenes"]},
        'DimensionResize' : {'dimension': 40, 'target': ["scenes","annotations"]},
        'ToTensor': {'target':["scenes","annotations"]}
    },
    'network': {
        'learning_rate': 0.001 ,
        'validation_split': 0.1 ,
        'momentum': 0.9 ,
        'randomized_seed': 10 ,
        'shuffle_dataset': False ,
        'epochs': 100 ,
        'NR_LSTM_UNITS': 2 ,
        'IMAGE_INPUT_SIZE_W': 640 ,
        'IMAGE_INPUT_SIZE_H': 480 ,
        'IMAGE_AFTER_CONV_SIZE_W': 18 ,
        'IMAGE_AFTER_CONV_SIZE_H': 13 ,
        'LSTM_IO_SIZE': 18 * 13 ,
        'LSTM_HIDDEN_SIZE': 18 * 13 ,
        'RGB_CHANNELS': 3 ,
        'TIMESTEPS': 40,  # size videos
        'BATCH_SIZE': 5  # until now just batch_size = 1
    },
    'logger':{
        'log_file': 'one_angle_test_1.log',
        'log_folder': '~/gait_project/logs',
        'plot_file': 'one_angle_test_1.png'
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

flows_40 = {
    'indexing':{
        'grouping': 'person_sequence_angle',
        'selection': 'manual_people',     #  => 'auto'= by final annotation or
                                 #  => 'manual_people' = uses 'people' list
                                 #  => 'manual_people_sequence' uses combination of two lists 'people' and 'sequences'
        'people_selection': [1,2,3,4,5,6,7,8,9,10,41,42,43,44,45,46,47,48,49,50],
        #'sequences_selection': ['nm-01']
        #'sequences_selection': ['bg-01','bg-02','cl-01','cl-02','nm-01','nm-02','nm-03','nm-04','nm-05','nm-06']
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
        'patch_size' : 5,
        'angles' : [54,90,126],
        'axis' : 1
        },
    'scenes':{
        'load':False,
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
    'dataset_output' : {
        'data': ["flows"],
        'label': "annotations"
    },
    'transformers':{
        # 'Crop':{'include list':['LAnkle','RAnkle'],'output_size':256,'target':'flows'}
        #'SpanImagesList': {'remove':False, 'names': ["heatmaps_LAnkle","heatmaps_RAnkle"],'target': ["heatmaps"]},
        # 'Rescale': {'output_size' : (480,640), 'target': ["flows"]},
        'AnnotationToLabel': {'target': ["annotations"]},
        'Transpose' : {'swapping': (2, 0, 1) , 'target': ["flows"]},
        'DimensionResize' : {'start':10,'dimension': 20, 'target': ["flows","annotations"]},
        'ToTensor': {'target':["flows","annotations"]},
        # 'Normalizer': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],'target':["flows"]}
    },
    'network': {
        'learning_rate': 0.01,
        'validation_split': 0.0,
        'momentum': 0.9,
        'randomized_seed': 10,
        'shuffle_dataset': False,
        'epochs': 1,
        'NR_LSTM_UNITS': 2 ,
        'IMAGE_INPUT_SIZE_W': 640 ,
        'IMAGE_INPUT_SIZE_H': 480 ,
        'IMAGE_AFTER_CONV_SIZE_W': 18 ,
        'IMAGE_AFTER_CONV_SIZE_H': 13 ,
        'LSTM_IO_SIZE': 58 * 13,
        'LSTM_HIDDEN_SIZE': 18 * 13,
        'RGB_CHANNELS': 3,
        'TIMESTEPS': 10,  # size videos
        'BATCH_SIZE': 5, # until now just batch_size = 1
        'device': "cuda:1"
    },
    'logger':{
        'log_file': 'flows_p20_lr0p01_bz5_ts10_control.log',
        'log_folder': '~/gait_project/logs',
        'plot_file': 'flows_p20_lr0p01_bz5_ts10_control.png',
        'model_file': 'flows_p20_lr0p01_bz5_ts10_control.tar'
    }
}



stack_flow_a = {

    'indexing':{
        'grouping': 'person_sequence_angle',
        'selection': 'manual_people',     #  => 'auto'= by final annotation or
                                 #  => 'manual_people' = uses 'people' list
                                 #  => 'manual_people_sequence' uses combination of two lists 'people' and 'sequences'
        'people_selection': [1,2,3,4,5,6,7,8,9,10,41,42,43,44,45,46,47,48,49,50],
        #'sequences_selection': ['nm-01']
        #'sequences_selection': ['bg-01','bg-02','cl-01','cl-02','nm-01','nm-02','nm-03','nm-04','nm-05','nm-06']
        },
    'pose': {
        'load': False,
        'preprocess': False,
        'D': 2 ,
        'body_keypoints_include_list': ['LAnkle','RAnkle','LKnee','RKnee','RHip','LHip']
        },
    'flow': {
        'load':True,
        'preprocess' : True,
        'method' : 'dense',
        'load_patches' : True,
        'patch_size' : 5,
        'angles' : [54,90,126],
        'axis' : 1
        },
    'scenes':{
        'load':False,
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
    'dataset_output' : {
        'data': ["flows"],
        'label': "annotations"
    },
    'transformers':{
        'AnnotationToLabel': {'target': ["annotations"]},
        'Transpose' : {'swapping': (2, 0, 1) , 'target': ["flows"]},
        'DimensionResize' : {'start':10,'dimension': 5, 'target': ["flows","annotations"]},
        'ToTensor': {'target':["flows","annotations"]},
    },
    'network': {
        'learning_rate': 0.01,
        'validation_split': 0.0,
        'momentum': 0.9,
        'randomized_seed': 10,
        'shuffle_dataset': False,
        'epochs': 1,
        'NR_LSTM_UNITS': 2 ,
        'IMAGE_INPUT_SIZE_W': 640 ,
        'IMAGE_INPUT_SIZE_H': 480 ,
        'IMAGE_AFTER_CONV_SIZE_W': 114 ,
        'IMAGE_AFTER_CONV_SIZE_H': 24 ,
        'LSTM_IO_SIZE': 24 * 114,
        'LSTM_HIDDEN_SIZE': 24 * 114,
        'RGB_CHANNELS': 3,
        'TIMESTEPS': 5,  # size videos
        'BATCH_SIZE': 5, # until now just batch_size = 1
        'device': "cuda:1"
    },
    'logger':{
        'log_file': 'flows_p20_lr0p01_bz5_ts10_control.log',
        'log_folder': '~/gait_project/logs',
        'plot_file': 'flows_p20_lr0p01_bz5_ts10_control.png',
        'model_file': 'flows_p20_lr0p01_bz5_ts10_control.tar'
    }
}

