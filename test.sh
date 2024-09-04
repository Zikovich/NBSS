#!/bin/bash

VERSION="version_36"

python SharedTrainer.py test --config=logs/OnlineSpatialNet/${VERSION}/config.yaml \
 --ckpt_path=logs/OnlineSpatialNet/${VERSION}/checkpoints/last.ckpt \
 --data.test_data_path=/teamspace/uploads/clarity_CEC3_data/task2/clarity_data/train/scenes \
 --data.test_json_file=test.json

#!/bin/bash