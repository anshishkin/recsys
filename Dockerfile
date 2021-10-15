# basic OS image
FROM ubuntu:20.04

# build-essentials and other necessary utils
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
    nano \
    && apt-get clean

RUN pip3 install -U pip
# project sources

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt && pip3 cache purge

RUN cd /usr && mkdir /src
WORKDIR /usr/src
COPY . .

#EXPOSE 8000
