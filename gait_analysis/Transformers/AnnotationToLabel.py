import pandas as pd
import numpy as np
import logging

class AnnotationToLabel(object):

    def __init__(self,parameters):
        self.target = parameters['target']
        self.logger = logging.getLogger()
    def __call__(self,sample):
        for target in self.target:
            annotations = sample[target]
            values = self.__annotation_to_labels(annotations)
            sample[target] = values

        return sample

    def __annotation_to_labels(self , annotations):
        # Create the combination of annotations
        values = []
        for left, right in zip(annotations["left_foot"],annotations["right_foot"]):
            if left =='IN_THE_AIR' and right=='ON_GROUND':
                values.append(0)
            elif left == 'ON_GROUND' and right=='IN_THE_AIR':
                values.append(1)
            elif left == 'ON_GROUND' and right == 'ON_GROUND':
                values.append(2)
            elif left == 'ON_THE_AIR' and right == 'ON_THE_AIR':
                self.logger.error('Error: double air. Please verify your annotations files.')
                self.logger.error('==================================')
                self.logger.error(annotations)

        return np.array(values)