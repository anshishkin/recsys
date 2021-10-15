# basic OS image
FROM python:3.8
  
# project sources
#COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install sklearn pandas numpy scipy mlflow s3 boto3 fastapi uvicorn[standart] argparse python-dotenv

COPY . /app
WORKDIR /app
EXPOSE 8000
CMD uvicorn main:app --host 0.0.0.0 --port 8000
#CMD python main.py
# SHELL ["/bin/bash", "-c"]
#ENTRYPOINT ['python3','test.py']