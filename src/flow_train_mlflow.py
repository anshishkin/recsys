
import os
import mlflow
import argparse
import shutil
from os import environ as env
import tempfile  

from dotenv import load_dotenv
load_dotenv()
from mlflow.pyfunc import PythonModel
from scipy.sparse.linalg import svds
import numpy as np 
from utils.load_yaml import load_yaml
from datalib.features.train import DataTrain
from model import SVD

from pathlib import Path
Path(__file__).parent

os.environ['AWS_ACCESS_KEY_ID'] = env['S3_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] =env['S3_SECRET_ACCESS_KEY']

class SVDModel(PythonModel):

    def __init__(self, n):
        self.n = n

    def predict(self, train_matrix):
        u, s, vt = svds(train_matrix, k=self.n)
        s_diag_matrix = np.diag(s)
        X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
        return X_pred

def mlflow_fit(project_name, check_mlproject=False):
    #mlflow_uri=load_yaml('s3_config.yml')
    mlflow.set_tracking_uri(env['MLFLOW_TRACKING_URI'])
    if check_mlproject==False:
        config= load_yaml('experiment.yml')
        param_id = config["exp_parametrs"]['n_com']
    else:
        param_id=None
    
    #name_exp = env['CI_JIRA_PROJECT_NAME'] #s3_conn['s3']['CI_JIRA_PROJECT_NAME']
    #print(config["exp_parametrs"]['name_exp'])
    service = mlflow.tracking.MlflowClient()
    model_train=DataTrain(SVD)
    for i in set(param_id):
        n_com = int(i)
        try:
            experiment_id = mlflow.create_experiment(name=project_name)
        except:
            experiment_id = mlflow.get_experiment_by_name(name=project_name).experiment_id

        mlflow_run = mlflow.start_run(experiment_id=experiment_id,run_name=f"run_with_param_{i}")
        run_id = mlflow.active_run().info.run_id    
        run = service.get_run(run_id)
        print("Metadata & data for run with UUID %s: %s" % (run_id, run))
        local_dir = tempfile.mkdtemp()
        try:
            file_path = os.path.join(local_dir, "exp_param.yaml")
            file_req = "requirements.txt"

            with mlflow_run:

                artifacts_path="model"
                model_fit, error_eval = model_train.fit_model(n_com=n_com)
                mlflow.log_param("n_com", n_com)
                model=SVDModel(n=n_com)
                mlflow.log_metric("RMSE", error_eval)
                mlflow.log_param("loss", 'regression')
                #mlflow.pyfunc.save_model(path="model_svd_new_4",python_model=self.model_, artifacts=artifacts)                   
                with open(file_path, "w") as handle:
                    handle.write(f"exp_param: \n n_com: {i} \n run_id: {run_id}")
                mlflow.log_artifact(file_path, artifact_path=artifacts_path)
                mlflow.pyfunc.log_model(python_model=model,artifact_path=artifacts_path,conda_env='conda.yaml')
                mlflow.log_artifact(file_req,artifact_path=artifacts_path)
                
                #mlflow.log_artifacts(local_dir, artifact_path=artifacts_path)
                #service.log_artifact(run_id=run_id,local_path='requirements.txt',artifact_path=artifacts_path)
                # if not os.path.exists("outputs"):
                #         os.makedirs("outputs")
                # with open("outputs/param.yaml", "w") as f:
                #     f.write(f"n_com: {i}")
                # mlflow.log_artifacts("outputs",artifact_path=artifacts_path)
                #mlflow.log_artifacts(local_dir, artifact_path=artifacts_path)
        finally:
            shutil.rmtree(local_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name')
    args = parser.parse_args()
    mlflow_fit(args.project_name,check_mlproject=False)