import numpy as np

class AbstractGaitModel():
    '''
    AbstractBaseModelClass
    '''

    def __init__(self):
        pass

    def train(self, X, Y):
        pass

    def validate(self, X, Y):
        pass


    def predict(self, X):
        return np.random.rand()

    def finish_training(self):
        pass
