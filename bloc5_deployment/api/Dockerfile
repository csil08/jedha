FROM continuumio/miniconda3

RUN apt-get update -y 
RUN apt-get install nano unzip curl -y

RUN conda install -y python=3.11 && conda clean -afy

RUN curl -fsSL https://get.deta.dev/cli.sh  |  sh

WORKDIR /home/app

COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

COPY . /home/app

CMD gunicorn app:app --bind 0.0.0.0:$PORT --worker-class uvicorn.workers.UvicornWorker #