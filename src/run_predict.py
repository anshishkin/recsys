from datalib.features.predict import DataPredict
from datalib.connection import EngineS3
import pickle
from utils import load_yaml
from pathlib import Path
Path(__file__).parent

class DataPredictSVD(DataPredict):
    
    def load_model_s3(self,exp_name):
            #s3_conn=load_yaml('s3_config.yml')
            s3=EngineS3().connection('resource')
            # model=pickle.loads(self.model)
            #bucket = 'kernelsvd'
            key = 'model/python_model.pkl'
            model = pickle.loads(
                s3.Bucket(exp_name).Object(key).get()['Body'].read())
            # s3.Object(bucket,key).put(Body=model)
            # s3.put_object(Body=model, Bucket='kernelsvd', Key='MovieLens')
            print("Model loaded .... Done! \n")
            return model
    def predict(self):
            config = load_yaml('production.yml')
            #runs_id = config["artifacts"]['run_id']
            exp_name=config["artifacts"]['exp_name']
            #print(runs_id)
            print(exp_name)
            #exp_id = mlflow.get_experiment_by_name(name=exp_name).experiment_id
            movie_list = movie_list()
            model=self.load_model_s3(exp_name)
            return model.predict(train_data_matrix), movie_list