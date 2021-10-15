import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.decomposition as skd
import pickle

from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.sparse.linalg import svds
import boto3
from botocore.client import Config

import mlflow


class DataLoad(object):

    def __init__(self):
        super().__init__()

    def loader_dataset(self):
        df = pd.read_csv('data/ml-latest-small/ratings.csv')
        self.input_list = df['movieId'].unique()
        return df

    def loader_dataset_s3(self):
        s3 = boto3.client('s3',
                          endpoint_url='http://192.168.42.113:9000',
                          aws_access_key_id='minio_admin',
                          aws_secret_access_key='minio_pass',
                          region_name='us-east-1',
                          config=Config(signature_version='s3v4')
                          )
        obj = s3.get_object(Bucket="kernelsvd", Key="MovieLens/ratings.csv")

        df = pd.read_csv(obj['Body'])
        self.input_list = df['movieId'].unique()
        return df

    def _scale_movie_id(self, input_id):
        return np.where(self.input_list == input_id)[0][0] + 1

    def dataset_load_s3(self):
        df = self.loader_dataset_s3()
        df['movieId'] = df['movieId'].apply(self._scale_movie_id)
        print("Dataset is loading ... Done! \n")
        return df


class DataPrepare(object):

    def __init__(self):
        self.DL = DataLoad()
        self.df = self.DL.dataset_load_s3()
        self.n_users = self.df['userId'].unique().shape[0]
        self.n_items = self.df['movieId'].unique().shape[0]

    def split_dataset(self, test_size: float = 0.2):
        train_data, test_data = train_test_split(
            self.df, test_size=test_size, random_state=0)
        return train_data, test_data

    def create_matrix(self):
        train_data, test_data = self.split_dataset()
        train_data_matrix = np.zeros((self.n_users, self.n_items))
        for line in train_data.itertuples():
            train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

        test_data_matrix = np.zeros((self.n_users, self.n_items))
        for line in test_data.itertuples():
            test_data_matrix[line[1] - 1, line[2] - 1] = line[3]
        # train_data_matrix = train_data.pivot(index='userId',
        #                                                   columns='movieId',
        #                                                   values='rating').fillna(0)
        # np.asarray(train_data_matrix)
        train_data_matrix_ = self.df.pivot(index='userId',
                                           columns='movieId',
                                           values='rating').fillna(0)

        test_data_matrix_ = self.df.pivot(index='userId',
                                          columns='movieId',
                                          values='rating').fillna(0)
        print("Dataset is preparing...Done! \n")
        return train_data_matrix, test_data_matrix, train_data_matrix_, test_data_matrix_


class DataTrain(object):
    def __init__(self):
        self.DP = DataPrepare()
        self.train_data_matrix, self.test_data_matrix, self.train_data_matrix_, self.test_data_matrix_ = self.DP.create_matrix()

    def _fit_model(self, n_com: int = 15):
        model_trsvd = skd.TruncatedSVD(n_components=n_com)
        self.model = model_trsvd.fit(self.train_data_matrix)

        u, s, vt = svds(self.train_data_matrix, k=15)
        s_diag_matrix = np.diag(s)

        # predict alternative
        X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
        self.error_eval = self.evaluate_model(X_pred, self.test_data_matrix)

        # error
        print('Error RMSE:', self.error_eval)
        self.save_model_s3()
        self.mlflow_track()
        #self.evaluate_model(self.model.transform(self.train_data_matrix),self.train_data_matrix)
        # self.X_transformed=self.model.transform(self.train_data_matrix)
        # self.VT = self.model.components_
        # self.U = self.X_transformed / self.model.singular_values_
        # self.sigma_matrix = np.diag(self.model.singular_values_)
        # print(self.VT,self.U,self.sigma_matrix)

    def evaluate_model(self, prediction, ground_truth):
        prediction = prediction[ground_truth.nonzero()].flatten()
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        return sqrt(mean_squared_error(prediction, ground_truth))

    def save_model(self):
        with open('model.pickle', 'wb') as f:
            pickle.dump(self.model, f)
        print("Model saved .... Done! \n")

    def save_model_s3(self):
        s3 = boto3.resource('s3',
                            endpoint_url='http://192.168.42.113:9000',
                            aws_access_key_id='minio_admin',
                            aws_secret_access_key='minio_pass',
                            region_name='us-east-1',
                            config=Config(signature_version='s3v4')
                            )

        model = pickle.dumps(self.model)
        bucket = 'kernelsvd'
        key = 'model.pkl'
        s3.Object(bucket, key).put(Body=model)
        # s3.put_object(Body=model, Bucket='kernelsvd', Key='MovieLens')
        print("Model saved .... Done! \n")

    def mlflow_track(self):      
        mlflow.set_tracking_uri("http://192.168.42.113:5000")
        try:
            experiment_id = mlflow.create_experiment(name="kernelsvd")
        except:
            experiment_id = mlflow.get_experiment_by_name(name="kernelsvd").experiment_id

        mlflow_run = mlflow.start_run(experiment_id=experiment_id, run_name="first")
        
        with mlflow_run:
            mlflow.log_metric("test_rmse", self.error_eval)

            mlflow.log_param("loss", 'regression')
            #mlflow.log_artifacts("Models", "SVD")

DataTrain()._fit_model()


class DataPredict(object):
    def __init__(self):
        self.DP = DataPrepare()
        self.train_data_matrix, self.test_data_matrix, self.train_data_matrix_, self.test_data_matrix_ = self.DP.create_matrix()

    def load_model(self):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model SVD load ...  Done \n")
        return model

    def movie_list(self):
        s3 = boto3.client('s3',
                          endpoint_url=os.getenv('AWS_S3_ENDPOINT_URL', None),
                          config=Config(signature_version='s3v4')
                          )
        mov = s3.get_object(Bucket="kernelsvd", Key="MovieLens/movies.csv")

        return pd.read_csv(mov['Body'])

    def load_model_s3(self):
        s3 = boto3.resource('s3',
                            endpoint_url=os.getenv('AWS_S3_ENDPOINT_URL', None),
                            config=Config(signature_version='s3v4')
                            )

        # model=pickle.loads(self.model)
        bucket = 'kernelsvd'
        key = 'model.pkl'
        model = pickle.loads(
            s3.Bucket(bucket).Object(key).get()['Body'].read())
        # s3.Object(bucket,key).put(Body=model)
        # s3.put_object(Body=model, Bucket='kernelsvd', Key='MovieLens')
        print("Model loaded .... Done! \n")
        return model

    def predict(self):
        model = self.load_model_s3()
        predictions = model.transform(self.test_data_matrix)
        print('Predict is done!')
        return predictions

    def predict_svd(self):
        movie_list = self.movie_list()
        u, s, vt = svds(self.train_data_matrix, k=15)
        s_diag_matrix = np.diag(s)
        # predict alternative
        X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
        print('Predict is done!')

        return X_pred, movie_list

#DataPredict().predict_svd()