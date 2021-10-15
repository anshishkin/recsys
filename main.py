from src.datalib.features.predict import DataPredict
from fastapi import FastAPI
import uvicorn
import numpy as np
from pydantic import BaseModel
from typing import Dict

class RequestClient(BaseModel):
    id_cl: int

class RequestTrain(BaseModel):
    train: bool

app = FastAPI()

@app.post('/predict')
def predict_rec_svd(data: RequestClient) -> Dict[str, np.ndarray]:
    data = data.dict()
    id_ = data['id_cl']
    pred, mv_list = DataPredict().predict()
    sort_mv = np.argsort(-pred[int(id_), :])[:15]
    return {f'Top 15 films recomendation for client_id = {id_}': mv_list[mv_list.movieId.isin(sort_mv)].iloc[:15, 1].tolist()}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
