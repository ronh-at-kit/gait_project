import torch

class Transpose(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,parameters):
        self.swapping = parameters['swapping']
        self.target = parameters['target']

        self.parameters = parameters

    def __call__(self, sample):
        for t in self.target:
            value = sample[t]
            if isinstance(value,list):
                v1 = value[0]
                sample[t] = [ v.transpose(self.swapping) for v in value]
            else:
                sample[t] = value.transpose(self.swapping)
        return sample