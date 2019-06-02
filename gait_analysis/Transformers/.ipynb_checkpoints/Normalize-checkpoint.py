import numpy as np
class Normalize(object):
    """Normalizes flow objects
    CAREFUL, expects [channels, height, width] or [channels, width, height]... TODO
    """

    def __init__(self, parameters):
        self.mean_flow = [126.04878374565972, 127.50835754123264, 1.5171344184027777]
        self.std_dev_flow = [7.03813454, 2.04901181, 5.66295848]
        self.target = parameters['target']

    def __call__(self,sample):
        for t in self.target:
            if t == 'flows':
                images = sample[t]
                if not isinstance(images, list):
                    flow_normalized = self.__normalize_flow(images)
                    sample[t] = flow_normalized
                else:
                    flow_normalized = [self.__normalize_flow(image) for image in images]
                    sample[t] = flow_normalized
            else:
                print("Warning! No normalization for this type specified")
        return sample

    def __normalize_flow(self,image):
        image = np.array(image, dtype=np.float64)
        # print("Normalizing image", image)
        image[0] = (image[0] - self.mean_flow[0]) / self.std_dev_flow[0]
        image[1] = (image[1] - self.mean_flow[1]) / self.std_dev_flow[1]
        image[2] = (image[2] - self.mean_flow[2]) / self.std_dev_flow[2]
        # print("Normalized image",image)
        return image
