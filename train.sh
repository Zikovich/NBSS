#!/bin/bash

python SharedTrainer.py fit \
  --config=configs/onlineSpatialNet.yaml \
  --config=configs/datasets/clarity_data_config.yaml \
  --model.channels="[0,1,2,3,4,5]" \
  --model.arch.dim_input=12 \
  --model.arch.dim_output=2 \
  --model.arch.num_freqs=129 \
  --trainer.precision=16-mixed \
  --model.compile=true \
  --data.batch_size="1" \
  --trainer.devices=1 \
  --trainer.max_epochs=200 \
  --model.loss.init_args.loss_func=models.io.loss.neg_si_sdr \
  --data.init_args.train_limit=1 \
  --data.init_args.val_limit=1 \
  --data.init_args.test_limit=1 \

#!/bin/bash
