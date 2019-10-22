import numpy as np
class DimensionResize(object):
    """Adjust Dimension
        makes all sizes equal length
    Args:
        dimension (int): Desired dimension.
    """

    def __init__(self, parameters):
        self.start = parameters['start'] if ('start' in parameters) else 0
        self.dimension = parameters['dimension']
        self.target = parameters['target']

    def __call__(self, sample):

        for t in self.target:
            vector = sample[t]
            # this vector is an indexation of frames per videos
            if len(vector) > self.dimension + self.start:
                vector = vector[self.start:self.start+self.dimension]
            elif len(vector) < self.dimension + self.start:
                vector = vector[self.start:-1]
                vector = self._repeat_value(vector, len(vector),self.start+self.dimension)
            sample[t] = vector

        return sample

    def _repeat_value(self, vector, start, stop):
        repeating_value = vector[-1]
        for i in range(start, stop):
            if isinstance(vector, list):
                vector.append(repeating_value)
            elif isinstance(vector, np.ndarray):
                vector = np.append(vector, repeating_value)
        return vector