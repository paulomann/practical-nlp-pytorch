#!/bin/bash
NV_GPU=$1 nvidia-docker run -tid --rm --shm-size=20g --ulimit memlock=-1 -v /home/paulomann/workspace/practical-nlp-pytorch:/workspace/practical-nlp practical-nlp:latest
