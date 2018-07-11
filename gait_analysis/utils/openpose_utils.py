import numpy as np
import json
from gait_analysis.utils.iterators import *

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

def maybe_subtract_poses(pose_keypoints, apply_diff_poses, convert_to_magnitude):
    '''
    If apply_diff_poses is True, the list of pose keypoints is returned
    that gives the difference between each consecutive keypoint frame.
    :param pose_keypoints:
    :param apply_diff_poses:
    :return:
    '''
    if apply_diff_poses:
        augmented_list = [pose_keypoints[0]] + pose_keypoints
        poses = []
        for p1, p2 in pairwise(augmented_list):
            diff_pose = Pose(p1) - Pose(p2)
            if convert_to_magnitude:
                poses.append(diff_pose.magnitudes)
            else:
                poses.append(diff_pose.to_list())
        return poses
    return pose_keypoints

class Pose():
    pose_keypoints = None
    def __init__(self, pose_keypoints):
        '''
        A class to easier perform calculations with pose keypoints
        :param pose_keypoints: a list of keypoints where each keypoint is a np array of length 2
        '''
        assert type(pose_keypoints) == list
        self.pose_keypoints = pose_keypoints

    def __sub__(pose1, pose2):
        diff_pose = []
        for arr1, arr2 in zip(pose1.pose_keypoints, pose2.pose_keypoints):
            diff_pose.append(arr1 - arr2)
        return Pose(diff_pose)

    def __str__(self):
        return self.pose_keypoints.__str__()

    @property
    def magnitudes(self):
        return [np.sqrt(arr[0]**2 + arr[1]**2) for arr in self.pose_keypoints]

    def to_list(self):
        return self.pose_keypoints


if __name__ == '__main__':
    pass
