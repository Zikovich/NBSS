#!/bin/bash

for VERSION in version_55 version_56 version_57 version_58; do
  echo "Running test for ${VERSION}"
  python SharedTrainer.py test --config=logs/OnlineSpatialNet/${VERSION}/config.yaml \
   --ckpt_path=logs/OnlineSpatialNet/${VERSION}/checkpoints/last.ckpt \
   --data.test_data_path=/teamspace/studios/this_studio/clarity_CEC3_data/task2/clarity_data/train/scenes \
   --data.test_json_file=test.json
done

#!/bin/bash