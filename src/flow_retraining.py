import os
from botocore.compat import copy_kwargs
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
import mlflow.pyfunc
import mlflow
import yaml
from os import environ as env
from dotenv import load_dotenv
import re
from pathlib import Path
Path(__file__).parent
load_dotenv()   
os.environ['AWS_ACCESS_KEY_ID'] = env['S3_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = env['S3_SECRET_ACCESS_KEY']


def load_yaml(name):
    config_path = os.path.join('configs/', name)
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

class DataLoad(object):

    def __init__(self):
        super().__init__()

    def loader_dataset(self):
        df = pd.read_csv('data/ml-latest-small/ratings.csv')
        self.input_list = df['movieId'].unique()
        return df

    def loader_dataset_s3(self):
        s3 = boto3.client('s3',
                          endpoint_url=env['S3_ENDPOINT_URL'],
                          aws_access_key_id=env['S3_ACCESS_KEY_ID'],
                          aws_secret_access_key=env['S3_SECRET_ACCESS_KEY'],
                          region_name='us-east-1',
                          config=Config(signature_version='s3v4')
                          )
        obj = s3.get_object(Bucket="kernelsvd", Key="data/ml-latest-small/ratings.csv")

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

class SVD(mlflow.pyfunc.PythonModel):

    def __init__(self, n):
        self.n = n

    def predict(self, train_matrix):
        u, s, vt = svds(train_matrix, k=self.n)
        s_diag_matrix = np.diag(s)
        # predict alternative
        X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
        return X_pred

class DataTrain(object):
    def __init__(self):
        self.DP = DataPrepare()
        self.train_data_matrix, self.test_data_matrix, self.train_data_matrix_, self.test_data_matrix_ = self.DP.create_matrix()

    def _fit_model(self,project_name,run_id):
        #parser = argparse.ArgumentParser()
        #parser.add_argument('--system-config',required=True)
        #args = parser.parse_args()
        #with open(args['system_config'], 'r') as file:
        #    system_config = json.load(file)
        #runs_path = system_config['runs_path']
        #parser.add_argument('--path',required=True)
        #parser.add_argument('--artifacts',required=True)
        mlflow.set_tracking_uri(env['MLFLOW_TRACKING_URI'])
        experiment_id = mlflow.get_experiment_by_name(name=f'{project_name}_exp').experiment_id
        config=self.load_artifacts_mlflow(experiment_id,run_id)
        
        self.model=SVD(n=config['exp_param']['n_com'])
        self.error_eval = self.evaluate_model(self.model.predict(self.train_data_matrix), self.test_data_matrix)
        self.mlflow_track(project_name)
        self.save_model_s3(project_name,experiment_id,run_id)
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

    def save_model_s3(self,project_name,exp_id,run_id):
        s3 = boto3.resource('s3',
                            endpoint_url=env['S3_ENDPOINT_URL'],
                            aws_access_key_id=env['S3_ACCESS_KEY_ID'],
                            aws_secret_access_key=env['S3_SECRET_ACCESS_KEY'],
                            region_name='us-east-1',
                            config=Config(signature_version='s3v4')
                            )
        # if not s3.Bucket(f'{project_name}') in s3.buckets.all():
        #     s3.create_bucket(Bucket=f'{project_name}')
        bucket_name=re.sub(r'_', '', project_name).lower()
        print(bucket_name)
        srcbucket = s3.Bucket('mlflow')
        try:
            destbucket = s3.create_bucket(Bucket=bucket_name,CreateBucketConfiguration={'LocationConstraint': 'us-east-1'})
        except:
            destbucket = s3.Bucket(bucket_name)
        
        src_prefix=f'{exp_id}/{run_id}/artifacts/model'
        dest_prefix='model'
        
        for obj in srcbucket.objects.filter(Prefix=src_prefix):
            src_source = {
                         'Bucket': 'mlflow',
                         'Key': obj.key
            #'Key': f'{exp_id}/{run_id}/artifacts/model/'
                          }
            new_key = obj.key.replace(src_prefix, dest_prefix, 1)
            new_obj = destbucket.Object(new_key)
            new_obj.copy(src_source)            
            print(obj.key +'- File Copied')

        # s3.meta.client.copy(copy_source, 'mlflow', f'{project_name}/model')
        # model = pickle.dumps(self.model)
        # #bucket = 'kernelsvd'
        # key = 'python_model.pkl'
        # s3.Object(exp_name, key).put(Body=model)
        # s3.put_object(Body=model, Bucket='kernelsvd', Key='MovieLens')
        print("Model saved! \n")
    
    def load_artifacts_mlflow(self, exp_id,run_id):
        #s3_conn=load_yaml('s3_config.yml')
        s3 = boto3.resource('s3',
                            endpoint_url=env['MLFLOW_S3_ENDPOINT_URL'],
                            aws_access_key_id=env['S3_ACCESS_KEY_ID'],
                            aws_secret_access_key=env['S3_SECRET_ACCESS_KEY'],
                            region_name='us-east-1',
                            config=Config(signature_version='s3v4')
                            )

        bucket = 'mlflow'
        key = f'{exp_id}/{run_id}/artifacts/model/exp_param.yaml'
        config = yaml.safe_load(
            s3.Bucket(bucket).Object(key).get()['Body'].read())
        #s3.Object(bucket, key).put(Body=model)
        # s3.put_object(Body=model, Bucket='kernelsvd', Key='MovieLens')
        return config

    def mlflow_track(self,project_name):      
        mlflow.set_tracking_uri(env['MLFLOW_TRACKING_URI'])
        try:
            experiment_id = mlflow.create_experiment(name=project_name)
        except:
            experiment_id = mlflow.get_experiment_by_name(name=project_name).experiment_id

        mlflow_run = mlflow.start_run(experiment_id=experiment_id, run_name="prod")
        
        with mlflow_run:
            #mlflow.log_param("n_com", n_com)
            mlflow.log_metric("test_rmse", self.error_eval)
            mlflow.log_param("loss", 'regression')
            #mlflow.pyfunc.save_model(path="model_svd_new_4", 
            #                        python_model=self.model_,
            #                        #artifacts=artifacts
            #                       )
            mlflow.pyfunc.log_model("model",python_model=self.model)


if __name__ == "__main__":
    config_prod=load_yaml('production.yml')
    DataTrain()._fit_model(
                            config_prod['artifacts']['exp_name'],
                            config_prod['artifacts']['run_id'],
                            )
