import numpy as np
class DimensionResize(object):
    """Adjust Dimension
        makes all sizes equal length
    Args:
        dimension (int): Desired dimension.
    """

    def __init__(self, parameters):

        self.dimension = parameters['dimension']
        self.target = parameters['target']

    def __call__(self, sample):
        for t in self.target:
            vector = sample[t]
            # this vector is an indexation of frames per videos
            if len(vector) > self.dimension:
                vector = vector[0:self.dimension]
            elif len(vector) < self.dimension:
                repeating_value = vector[-1]
                for i in range(len(vector),self.dimension):
                    if isinstance(vector,list):
                        vector.append(repeating_value)
                    elif isinstance(vector,np.ndarray):
                        vector = np.append(vector,repeating_value)

            sample[t] = vector

        return sample