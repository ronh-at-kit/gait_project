import numpy as np


class Normpose(object):
    """Normalizes flow objects
    CAREFUL, expects [channels, height, width] or [channels, width, height]... TODO
    """

    def __init__(self, parameters):
        self.im_width = 640
        self.im_height = 480

        self.target = parameters['target']

    def __call__(self, sample):
        for t in self.target:
            if t == 'poses':
                poses = sample[t]

                norm_vec = np.array([self.im_width, self.im_height])
                poses_normalized = [pose / norm_vec for pose in poses]
                sample[t] = poses_normalized

            else:
                print("Warning! No normalization for this type specified")
        return sample




