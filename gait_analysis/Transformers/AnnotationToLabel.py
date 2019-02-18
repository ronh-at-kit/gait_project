import pandas as pd
import numpy as np


class AnnotationToLabel(object):

    def __init__(self,parameters):
        self.target = parameters['target']
    def __call__(self,sample):
        for target in self.target:
            annotations = sample[target]
            values = self.__annotation_to_labels(annotations)
            sample[target] = values

        return sample

    def __annotation_to_labels(self , annotations):
        # Create the combination of annotations
        annotations['combined'] = annotations["left_foot"] + annotations["right_foot"]
        # Convert into categorical type of data
        annotations.combined = pd.Categorical(annotations.combined)
        # Capture the codes from the categories
        annotations['codes'] = annotations.combined.cat.codes
        # update the annotations in the sample as numeric categories
        values = annotations.codes.values.astype(np.long)
        return values