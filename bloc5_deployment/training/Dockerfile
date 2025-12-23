FROM continuumio/miniconda3

WORKDIR /home/app

RUN conda install -y python=3.11 && conda clean -afy

RUN apt-get update
RUN apt-get install nano unzip
RUN apt install curl -y

RUN curl -fsSL https://get.deta.dev/cli.sh | sh

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

COPY train.py /home/app/train.py

CMD ["bash", "-c", "\
mlflow server \
  --host 0.0.0.0 \
  --port $PORT \
  --backend-store-uri $BACKEND_STORE_URI \
  --default-artifact-root $ARTIFACT_STORE_URI \
  --serve-artifacts"]