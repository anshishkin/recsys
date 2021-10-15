import pandas as pd
import unittest
import os, sys
from pathlib import Path
import yaml
import boto3
from botocore.client import Config
from os import environ as env
from dotenv import load_dotenv

load_dotenv()


THIS_DIR = Path(__file__).parent

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = THIS_DIR.parent / 'data/ml-latest-small/ratings.csv'
colnames = ["userId","movieId","rating","timestamp"]
id = set([1, 2, 3])

def load_yaml(name):
    config_path = os.path.join('configs/', name)
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def loader_dataset_s3():
    s3_conn=load_yaml('s3_config.yml')
    
    s3 = boto3.client('s3',
                        endpoint_url="http://192.168.42.113:9000",
                        aws_access_key_id="minio_admin",
                        aws_secret_access_key="minio_pass",
                        region_name="us-east-1",
                        config=Config(signature_version='s3v4')
                        )
    obj = s3.get_object(Bucket="kernelsvd", Key="MovieLens/ratings.csv")

    df = pd.read_csv(obj['Body'])
    return df


class DfTests(unittest.TestCase):
    def setUp(self):
        try:
            data = loader_dataset_s3()
            self.fixture = data
        except IOError as e:
            print(e)

    def test_colnames(self):
        self.assertListEqual(list(self.fixture.columns), colnames)

    # def test_timestamp_format(self):
    #     ts = self.fixture["timestamp"]
    #     # You need to check for every row in the dataframe
    #     [self.assertRegex(i, r"\d{2}-\d{2}-\d{4}") for i in ts]

    def test_user(self):
        df_userid = self.fixture["userId"]
        self.assertTrue(any([i in id for i in df_userid]))


if __name__ == "__main__":
    unittest.main()

# import unittest
# import pandas as pd
# from pandas.util.testing import assert_frame_equal # <-- for testing dataframes

# class DFTests(unittest.TestCase):

#     """ class for running unittests """

#     def setUp(self):
#         """ Your setUp """
#         try:
#             data = pd.read_csv('../data/ml-latest-small/ratings.csv',
#                 sep = ',',
#                 header = True)
#         except IOError:
#             print('cannot open file')
#         self.fixture = data

#     def test_dataFrame_constructedAsExpected(self):
#         """ Test that the dataframe read in equals what you expect"""
#         foo = pd.DataFrame()
#         assert_frame_equal(self.fixture, foo)

# if __name__ == '__main__':
#     unittest.main()