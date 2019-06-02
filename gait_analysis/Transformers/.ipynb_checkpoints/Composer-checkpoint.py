from gait_analysis.Config import Config
from torchvision import transforms

class Composer():
    def __init__(self):
        self.config = Config()

    def compose(self):
        '''
        Creates a transformes using the configuration.py
        :return:
        '''
        dataloader_configs = self.config.config['transformers']
        transformers = []
        for k in dataloader_configs.keys():
            parameters = dataloader_configs.get(k)
            module = __import__('gait_analysis.Transformers.' + k)
            class_ = getattr(module , k)
            transformers.append(class_(parameters))
        transform = transforms.Compose(transformers)
        return transform
