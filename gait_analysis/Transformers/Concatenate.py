import numpy as np
class Concatenate(object):
    """Concatenate images to channels (RGB-dimension)
        resulting output: [height, width, channels]
        resulting label has to be specified (e.g. middle or last frame)
    Args:
        quantity (int): Number images to concatenate
    """

    def __init__(self, parameters):
        self.quantity = parameters['quantity']
        if parameters['pos_label'] > self.quantity or parameters['pos_label'] < 0:
            self.pos_label = parameters['quantity']
        else:
            self.pos_label = parameters['pos_label']
        self.target = parameters['target']

    def __call__(self,sample):
        for t in self.target:
            if t == 'annotations':
                # print("Annotation:",sample[t])
                label = sample[t]
                label = label[self.pos_label]
                sample[t] = np.array(np.expand_dims(label, axis=0))
            else:
                image_list = sample[t]
                # print("Flow size:", len(image_list))
                image_list_new = np.concatenate(image_list,axis=0)
                # print(image_list_new.shape)
                sample[t] = image_list_new
        return sample
