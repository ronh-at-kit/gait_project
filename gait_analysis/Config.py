import gait_analysis.settings as settings
import gait_analysis.configurations as configurations
class Config():
    """
        If we need more configuration modes we can change the behavior of the class to load the proper
        configuration from the config.py or the tum_config.
    """
    def __init__(self ):
        configuration = settings.configuration
        if configuration == 'default': # more configuration schemes could come later
            self.config = configurations.default
        # TODO: create a tum_gait Config when refactor
        # else if configuration == 'tum'
        #     self.config = configurations.tum
        else:
            # if not specified select default
            self.config = configurations.default
