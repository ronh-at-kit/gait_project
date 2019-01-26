import torch

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,parameters):
        self.target = parameters['target']

        self.parameters = parameters

    def __call__(self, sample):
        for t in self.target:
            value = sample[t]
            if isinstance(value,list):
                sample[t] = [ torch.from_numpy(v) for v in value]
            else:
                sample[t] = torch.from_numpy(value)
        return sample