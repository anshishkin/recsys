import numpy as np
import pandas as pd

from ..connection import EngineS3


class DataLoad():

    def __init__(self):
        super().__init__()

    def loader_dataset(self):
        df = pd.read_csv('data/ml-latest-small/ratings.csv')
        self.input_list = df['movieId'].unique()
        return df

    def loader_dataset_s3(self):
        #s3_conn=load_yaml('s3_config.yml')
        s3=EngineS3().connection('client')
        
        obj = s3.get_object(Bucket="kernelsvd", Key="MovieLens/ratings.csv")

        df = pd.read_csv(obj['Body'])
        self.input_list = df['movieId'].unique()
        return df

    def _scale_movie_id(self, input_id):
        return np.where(self.input_list == input_id)[0][0] + 1

    def dataset_load_s3(self):
        df = self.loader_dataset_s3()
        df['movieId'] = df['movieId'].apply(self._scale_movie_id)
        print("Dataset is loading\n")
        return df