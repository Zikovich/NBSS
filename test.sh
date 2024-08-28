#!/bin/bash

python SharedTrainer.py test --config=logs/OnlineSpatialNet/version_47/config.yaml \
 --ckpt_path=logs/OnlineSpatialNet/version_47/checkpoints/last.ckpt \
 --data.test_data_path=/teamspace/studios/this_studio/clarity_CEC3_data/task2/clarity_data/train/scenes \
 --data.test_json_file=test.json



#!/bin/bash