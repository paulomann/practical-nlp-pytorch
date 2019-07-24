FROM nvidia/cuda:9.0-base-ubuntu16.04
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get install -y wget tar gzip nano bzip2
WORKDIR /workspace/
RUN wget https://repo.continuum.io/archive/Anaconda3-2019.03-Linux-x86_64.sh -O /workspace/conda.sh
RUN bash /workspace/conda.sh -b -p /workspace/anaconda3
RUN rm conda.sh
ENV PATH="/workspace/anaconda3/bin:$PATH"
RUN /bin/bash -c "conda update --prefix /workspace/anaconda3 anaconda"
RUN /bin/bash -c "conda init"
RUN ["/bin/bash", "-c", "pip install torch torchvision"]
RUN ["/bin/bash", "-c", "pip install spacy"]
RUN ["/bin/bash", "-c", "python -m spacy download pt_core_news_sm"]
RUN ["/bin/bash", "-c", "python -m spacy download en_core_web_sm"]
RUN ["/bin/bash", "-c", "pip install plotly==4.0.0"]
