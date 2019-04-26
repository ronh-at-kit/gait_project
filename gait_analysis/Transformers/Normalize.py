import numpy as np
class Normalize(object):
    """Normalizes flow objects
    CAREFUL, expects [channels, height, width] or [channels, width, height]... TODO
    """

    def __init__(self, parameters):
        self.mean_flow = [126.1972, 127.50477, 1.3415707]
        self.std_dev_flow = [6.436958,  1.7359118, 5.022056 ]
        self.mean_scenes = [104.41519, 124.919014, 95.53079]
        self.std_dev_scenes = [23.682524, 29.465359, 22.749443]

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
                    
            elif t == 'scenes':
                images = sample[t]
                if not isinstance(images, list):
                    scene_normalized = self.__normalize_scenes(images)
                    sample[t] = scene_normalized
                else:
                    scene_normalized = [self.__normalize_scenes(image) for image in images]
                    sample[t] = scene_normalized

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
    
    def __normalize_scenes(self,image):
        image = np.array(image, dtype=np.float64)
        # print("Normalizing image", image)
        image[0] = (image[0] - self.mean_scenes[0]) / self.std_dev_scenes[0]
        image[1] = (image[1] - self.mean_scenes[1]) / self.std_dev_scenes[1]
        image[2] = (image[2] - self.mean_scenes[2]) / self.std_dev_scenes[2]
        # print("Normalized image",image)
        return image
