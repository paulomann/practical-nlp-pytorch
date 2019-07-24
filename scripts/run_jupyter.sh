#!/bin/bash
NV_GPU=$1 nvidia-docker run -tid --rm --shm-size=20g --ulimit memlock=-1 -v /home/paulomann/workspace/practical-nlp-pytorch:/workspace/practical-nlp -p $2:$2 practical-nlp:latest
containerId=$(docker ps | grep 'practical-nlp:latest' | awk '{ print $1 }')
docker exec -ti $containerId bash -c 'cd /workspace/practical-nlp;cd notebooks;jupyter notebook --ip=0.0.0.0 --no-browser --port '"$2"' --allow-root'
