class SpanImagesList(object):
    def __init__(self,parameters):
        if not 'remove' in parameters:
            self.remove = False
        else:
            self.remove = parameters['remove']

        if not 'names' in parameters:
            self.names = None
        else:
            self.names = parameters['names']
        self.target = parameters['target']
    def __call__(self,sample):
        for target in self.target:
            # these are lists indexed by frames in a video
            # each frame has two images stores in a list...
            # TODO: MORE APPROPRIATED INDEXING IS BY FRAME???
            lists_images_list = sample[target]
            # initialize empty list to fill by frame
            for name in self.names:
                sample[name] = []

            for list_images_list in lists_images_list: # iteration over frames
                # list_images_list is a list that contains list of images
                for name , image in zip(self.names , list_images_list):
                    sample[name].append(image)

            if self.remove:
                sample.pop(target,None)
        return sample


    def __merge_two_dicts(x, y):
        z = x.copy()   # start with x's keys and values
        z.update(y)    # modifies z with y's keys and values & returns None
        return z