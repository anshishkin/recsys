import boto3
from botocore.client import Config
from os import environ as env
from dotenv import load_dotenv
import os

class EngineS3():
    def __init__(self):
        load_dotenv()   
    
    def connection(self,method:str):
        
        if method=='client':
            s3 = boto3.client('s3',
                    endpoint_url=env['S3_ENDPOINT_URL'],
                    aws_access_key_id=env['S3_ACCESS_KEY_ID'],
                    aws_secret_access_key=env['S3_SECRET_ACCESS_KEY'],
                    region_name='us-east-1',
                    config=Config(signature_version='s3v4')
                    )
        else:
            s3 = boto3.resource('s3',
                    endpoint_url=env['S3_ENDPOINT_URL'],
                    aws_access_key_id=env['S3_ACCESS_KEY_ID'],
                    aws_secret_access_key=env['S3_SECRET_ACCESS_KEY'],
                    region_name='us-east-1',
                    config=Config(signature_version='s3v4')
                    )
        return s3