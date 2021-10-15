import numpy as np
from .prepare import DataPrepare
from math import sqrt
from sklearn.metrics import mean_squared_error

class DataTrain():
    def __init__(self,base_model:object):
        self.DP = DataPrepare()
        self.train_data_matrix, self.test_data_matrix, self.train_data_matrix_, self.test_data_matrix_ = self.DP.create_matrix()
        self.base_model=base_model

    
    def fit_model(self, n_com: int = 15):
        model=self.base_model(n_com)
        error_eval = self.evaluate_model(self.base_model(n_com).predict(self.train_data_matrix), self.test_data_matrix)
        return model, error_eval
    
    def evaluate_model(self, prediction, ground_truth):
        prediction = prediction[ground_truth.nonzero()].flatten()
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        return sqrt(mean_squared_error(prediction, ground_truth))