import os
import numpy as np
import pandas as pd
import pickle
import mlflow
from os import environ as env
from dotenv import load_dotenv
import argparse
load_dotenv()
import re
from .prepare import DataPrepare
from ..data_utils import load_yaml
from ..connection import EngineS3

from pathlib import Path
Path(__file__).parent.parent

class DataPredict():

    def __init__(self):
        self.DP = DataPrepare()
        self.train_data_matrix, self.test_data_matrix, self.train_data_matrix_, self.test_data_matrix_ = self.DP.create_matrix()

    def load_model(self):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model SVD load ...  Done \n")
        return model

    def movie_list(self):
        #s3_conn=load_yaml('s3_config.yml')
        s3=EngineS3().connection('client')
        mov = s3.get_object(Bucket="kernelsvd", Key="data/ml-latest-small/movies.csv")

        return pd.read_csv(mov['Body'])

    def load_model_s3(self, exp_name):
        #s3_conn=load_yaml('s3_config.yml')
        s3=EngineS3().connection('resource')
        # model=pickle.loads(self.model)
        bucket = re.sub(r'_', '', exp_name).lower()
        print(bucket)
        key = 'model/python_model.pkl'
        #model=mlflow.pyfunc.load_model(f'{s3}://{bucket}/{key}')
        model = pickle.loads(s3.Bucket(bucket).Object(key).get()['Body'].read())
        # with open(python_model.pkl', 'wb') as data:
        #     s3.Bucket(bucket).download_fileobj(key, data)

        # with open('python_model.pkl', 'rb') as data:
        #     model = pickle.load(data)
        print("Model loaded .... Done! \n")
        return model
    
    def load_model_mlflow(self, exp_id,run_id):
        #s3_conn=load_yaml('s3_config.yml')
        s3=EngineS3().connection('resource')
        bucket = 'mlflow'
        key = f'{exp_id}/{run_id}/artifacts/model/python_model.pkl'
        model = pickle.loads(
            s3.Bucket(bucket).Object(key).get()['Body'].read())
        #s3.Object(bucket, key).put(Body=model)
        # s3.put_object(Body=model, Bucket='kernelsvd', Key='MovieLens')
        return model

    def predict(self):
        config = load_yaml('production.yml')
        #runs_id = config["artifacts"]['run_id']
        exp_name=config["artifacts"]['exp_name']
        #print(runs_id)
        print(exp_name)
        #exp_id = mlflow.get_experiment_by_name(name=exp_name).experiment_id
        movie_list = self.movie_list()
        model=self.load_model_s3(exp_name)
        return model.predict(self.train_data_matrix), movie_list

    def predict_svd(self):
        #with open('configs/system.json', 'r') as file:
        #     system_config = json.load(file)
        #runs_path = system_config['runs_path']
        #if 'MLFLOW_ID' in os.environ:
        
        #runs_path=env['MLFLOW_ID']
        #print(env['AWS_SECRET_ACCESS_KEY'])
        #if 'MLFLOW_ID' in os.environ:
            #parser = argparse.ArgumentParser()
            #parser.add_argument('--experiments', default='production.yml')
            #args = parser.parse_args()
            
        config=load_yaml('production.yml')
        runs_id = config["artifacts"]['run_id']
        exp_name=config["artifacts"]['exp_name']
        print(runs_id)
        print(exp_name)
        exp_id = mlflow.get_experiment_by_name(name=f"{exp_name}_exp").experiment_id
        movie_list = self.movie_list()
        model=self.load_model_mlflow(exp_id,runs_id)
        return model.predict(self.train_data_matrix), movie_list
        