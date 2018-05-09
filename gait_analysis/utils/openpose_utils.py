import numpy as np
import json

keypoints_mapping_coco_18 = {
    0:  "Nose",
    1:  "Neck",
    2:  "RShoulder",
    3:  "RElbow",
    4:  "RWrist",
    5:  "LShoulder",
    6:  "LElbow",
    7:  "LWrist",
    8:  "RHip",
    9:  "RKnee",
    10: "RAnkle",
    11: "LHip",
    12: "LKnee",
    13: "LAnkle",
    14: "REye",
    15: "LEye",
    16: "REar",
    17: "LEar",
    #18: "Background" not used
}

def load_keypoints_from_file(fname):
    '''
    returns the plain keypoints file as a dictionary
    :param fname:
    :return:
    '''
    with open(fname, 'r') as f:
        keypoints = json.load(f)
    return keypoints

def filter_keypoints(pose_dict, include_list, return_list=False, return_confidence=True):
    assert type(include_list) is list, 'include_list hast to be a list'
    output = {key: val for key, val in pose_dict.iteritems() if key in include_list}
    if not return_confidence:
        output = {key: val[:2] for key, val in output.iteritems()}
    if return_list:
        return output.values()
    return  output

def keypoints_to_posedict(pose):
    '''
    data format for each keypoint is x,y, confidence
    :param pose:
    :return:
    '''
    pose = np.array(pose).reshape(-1, 3)
    pose_dict = {point: pose[i, :].squeeze() for i, point in keypoints_mapping_coco_18.iteritems()}
    return pose_dict

def random_rgb():
    '''
    returns a random RGB array
    :return:
    '''
    return np.random.randint(0, 256, size=3)

