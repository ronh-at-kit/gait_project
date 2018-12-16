import gait_analysis.settings as settings
import gait_analysis.configurations as configurations
from gait_analysis.utils.decorators import singleton

@singleton
class Config(object):
    """
        If we need more configuration modes we can change the behavior of the class to load the proper
        configuration from the configurations.py and settings.py

    """
    def __init__(self ):
        configuration = settings.configuration
        print('loading configuration ', configuration)
        if configuration == 'default': # more configuration schemes could come later
            self.config = configurations.default
        # TODO: create a tum_gait Config when refactor
        # else if configuration == 'tum'
        #     self.config = configurations.tum
        else:
            # if not specified select default
            self.config = configurations.default
if __name__ == '__main__':
    c = Config()
    print(c.config)
    c.config['pose']['load'] = False
    c2 = Config()
    c3 = Config()
    print(c3.config)

