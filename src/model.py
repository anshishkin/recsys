from mlflow.pyfunc import PythonModel
from scipy.sparse.linalg import svds
import numpy as np

class SVD(PythonModel):

    def __init__(self, n):
        self.n = n

    def predict(self, train_matrix):
        u, s, vt = svds(train_matrix, k=self.n)
        s_diag_matrix = np.diag(s)
        X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
        return X_pred