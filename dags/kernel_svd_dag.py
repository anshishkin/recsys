import sys
import os
import yaml
from airflow import DAG
from datetime import timedelta
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.providers.amazon.aws.sensors.s3_key import S3KeySensor


this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_dir)

# Load configs
config_path = os.path.join(this_dir, 'config.yml')
with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)

# Set dag name
branch_name = config["branch_name"]
project_name = config["project_name"]
registry_image = config["registry_image"]
dag_name = f'{project_name}_{branch_name}'


default_args = {
    "start_date": "2021-4-20",
    "email": ["a.shishkin@talmer.ru"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5)
}


with DAG(dag_name, tags=[project_name], default_args=default_args, schedule_interval='*/15 * * * *', catchup=False) as dag:

    start = DummyOperator(task_id="start")

    task_sensor = S3KeySensor(task_id='sensor_s3',
                              bucket_key='data/ml-latest-small/ratings.csv',
                              wildcard_match=True,
                              bucket_name='kernelsvd',
                              aws_conn_id='conn_S3',
                              timeout=60 * 120,
                              poke_interval=10
                              )

    task_docker = DockerOperator(
        task_id='task_docker',
        image=f'{registry_image}:{branch_name}',
        api_version='auto',
        auto_remove=True,
        do_xcom_push=False,
        command='python3 src/flow_retraining.py',
        docker_conn_id="docker_gitlab",
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge'
        # container_name='kernel-svd_kernel_svd_dev_1'
    )

    end = DummyOperator(task_id="end")

start >> task_sensor >> task_docker >> end
