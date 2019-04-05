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
                print("Annotation:",sample[t])
                label = sample[t]
                label = label[self.pos_label]
                print("Label:", label)
                # e_arr = np.zeros(1, dtype=np.int64)
                # print("E-arr", e_arr)
                # e_arr[0] = label
                # print("New E arr", e_arr)
                # sample[t] = e_arr
                sample[t] = np.array([label])
            else:
                image_list = sample[t]
                print("Flow size:", len(image_list))
                image_list_new = np.concatenate(image_list,axis=2)
                print(image_list_new.shape)
                # e_arr = np.zeros(1, dtype=np.int64)
                # e_arr[0] = image_list_new
                sample[t] = image_list_new
                # print("Sample image: ",sample[t])
            # flat_sample = [item for sublist in sample for item in sublist]
        return sample
    # def __call__(self, sample):
    #     for t in self.target:
    #         vector = sample[t]
    #         # this vector is an indexation of frames per videos
    #         if len(vector) > self.dimension + self.start:
    #             vector = vector[self.start:self.start+self.dimension]
    #         elif len(vector) < self.dimension + self.start:
    #             vector = vector[self.start:-1]
    #             vector = self._repeat_value(vector, len(vector),self.start+self.dimension)
    #         sample[t] = vector
    #     return sample

    # def __call__(self, sample):
    #     for t in self.target:
    #         value = sample[t]
    #         if isinstance(value,list):
    #             sample[t] = [ torch.from_numpy(v) for v in value]
    #         else:
    #             sample[t] = torch.from_numpy(value)
    #     return sample
