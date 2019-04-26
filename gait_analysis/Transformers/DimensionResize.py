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
        self.offset = parameters['annotations_offset'] if ('annotations_offset' in parameters) else 0

    def __call__(self, sample):
        for t in self.target:
            if t == 'annotations':
                vector = sample[t]
                # this vector is an indexation of frames per videos
                if len(vector) >= self.dimension + self.start + self.offset:
                    vector = vector[self.start + self.offset:self.start+self.dimension+self.offset]
                else: # len(vector) < self.dimension + self.start + self.offset:
                    vector = vector[self.start+self.offset:-1]
                    vector = self._repeat_value(vector, len(vector),self.start+self.dimension + self.offset)
                sample[t] = vector
            else: #original
                vector = sample[t]
                # this vector is an indexation of frames per videos
                if len(vector) >= self.dimension + self.start:
                    vector = vector[self.start:self.start + self.dimension]
                else: # len(vector) < self.dimension + self.start:
                    vector = vector[self.start:-1]
                    vector = self._repeat_value(vector, len(vector), self.start + self.dimension)
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


# for t in self.target:
#     vector = sample[t]
#     # this vector is an indexation of frames per videos
#     if len(vector) > self.dimension + self.start:
#         vector = vector[self.start:self.start + self.dimension]
#     elif len(vector) < self.dimension + self.start:
#         vector = vector[self.start:-1]
#         vector = self._repeat_value(vector, len(vector), self.start + self.dimension)
#     sample[t] = vector