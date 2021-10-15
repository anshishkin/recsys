import numpy as np
from .load import DataLoad
from sklearn.model_selection import train_test_split

class DataPrepare():

    def __init__(self):
        self.DL = DataLoad()
        self.df = self.DL.dataset_load_s3()


    def split_dataset(self, test_size: float = 0.2):
        train_data, test_data = train_test_split(
            self.df, test_size=test_size, random_state=0)
        return train_data, test_data

    def create_matrix(self):
        train_data, test_data = self.split_dataset()
        n_users = self.df['userId'].unique().shape[0]
        n_items = self.df['movieId'].unique().shape[0]
        train_data_matrix = np.zeros((n_users, n_items))
        for line in train_data.itertuples():
            train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

        test_data_matrix = np.zeros((n_users, n_items))
        for line in test_data.itertuples():
            test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

        train_data_matrix_ = self.df.pivot(index='userId',
                                           columns='movieId',
                                           values='rating').fillna(0)

        test_data_matrix_ = self.df.pivot(index='userId',
                                          columns='movieId',
                                          values='rating').fillna(0)
        print("Dataset is preparing\n")
        return train_data_matrix, test_data_matrix, train_data_matrix_, test_data_matrix_