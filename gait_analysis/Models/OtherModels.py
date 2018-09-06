from BaseModels import AbstractGaitModel
import numpy as np

class SVMModel(AbstractGaitModel):
    '''
    uses sklearn convention
    '''
    feature_list = None
    target_list = None
    svm_obj = None
    is_trained = False
    def __init__(self, SVM):
        self.svm_obj = SVM
        self.feature_list = []
        self.target_list = []
        self.is_trained = False

    def train(self, X, Y):
        '''
        train on features
        :param X:
        :param Y:
        :param X: array-like, shape (n_samples, n_features)
        :param Y: array-like, shape (n_samples,)
        Target values (class labels in classification, real numbers in regression)
        :return:
        '''
        self.feature_list.append(X)
        self.target_list.append(Y)

    def validate(self, X, Y):
        pass


    def finish_training(self):
        self._concat_features()
        self.svm_obj.fit(self.feature_list, self.target_list)
        self.is_trained = True

    def _concat_features(self):
        if not self.is_trained:
            self.feature_list = np.vstack(self.feature_list)
            self.target_list = np.vstack(self.target_list).squeeze()
            assert self.feature_list.shape[0] == self.target_list.shape[0]
            assert len(self.target_list.shape) == 1
        else:
            pass


    def reset(self):
        self.feature_list = []
        self.feature_list = []

    def predict(self, X):
        '''
        Perform classification on samples in X.
        :param X:
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        :return:
        '''
        return self.svm_obj.predict(X)



