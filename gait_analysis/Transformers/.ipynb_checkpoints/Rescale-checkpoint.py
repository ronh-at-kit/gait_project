from cv2 import resize
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, parameters):
        output_size = parameters['output_size']
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.target = parameters['target']

    def __call__(self, sample):
        # TODO: TRY OUT IN THE HEATMAPS
        for t in self.target:
            images = sample[t]
            if not isinstance(images,list):
                image_resized = self.__rescale()
                sample[t] = image_resized
            else:
                images_resized = [self.__rescale(image) for image in images]
                sample[t] = images_resized
        return sample

    def __rescale(self,image):
        h , w = image.shape[:2]
        if isinstance(self.output_size , int):
            if h > w:
                new_h , new_w = self.output_size * h / w , self.output_size
            else:
                new_h , new_w = self.output_size , self.output_size * w / h
        else:
            new_h , new_w = self.output_size
        new_h , new_w = int(new_h) , int(new_w)
        image_resized = resize(image , (new_w , new_h))
        return image_resized