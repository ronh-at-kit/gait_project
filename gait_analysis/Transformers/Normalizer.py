import torch
from torchvision import transforms
class Normalizer(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,parameters):
        self.mean = parameters['mean']
        self.std = parameters['std']
        # parameters = {'mean': [0.485 , 0.456 , 0.406] , 'std': [0.229 , 0.224 , 0.225]}
        normalize = transforms.Normalize(mean=parameters['mean'] , std=parameters['std'])
        self.transform = transforms.Compose([
            transforms.ToPILImage() ,
            transforms.ToTensor() ,
            normalize
        ])
        self.target = parameters['target']

    def __call__(self, sample):
        for t in self.target:
            value = sample[t]
            if isinstance(value,list):
                sample[t] = [ self.transform(v) for v in value]
            else:
                sample[t] = self.transform(sample[t])
        return sample