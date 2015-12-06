#!/bin/bash

export APOLLO_ROOT=/home/jda/3p/apollocaffe
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$APOLLO_ROOT/build/lib
export PYTHONPATH=$PYTHONPATH:$APOLLO_ROOT/python:$APOLLO_ROOT/python/caffe/proto

#python main.py -c config/vqa_att.yml
python main.py -c config/vqa_nmn.yml
#kernprof -l main.py -c config/cocoqa_nmn.yml
