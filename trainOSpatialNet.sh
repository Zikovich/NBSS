#!/bin/bash

python SharedTrainer.py fit \
  --config=configs/onlineSpatialNet.yaml \
  --config=configs/datasets/clarity_data_config.yaml \
  --model.channels="[0,1,2,3,4,5]" \
  --model.arch.dim_input=12 \
  --model.arch.dim_output=2 \
  --model.arch.num_freqs=129 \
  --model.optimizer="[Adam, {lr: 0.001, weight_decay: 0.001}]" \
  --trainer.precision=16-mixed \
  --model.compile=true \
  --data.batch_size="4" \  # As stated in paper Changsheng Quan
  --data.train_limit="0" \
  --data.val_limit="0" \  # Limit validation set to 100 samples, None for full data
  --data.test_limit="0" \
  --trainer.devices=1 \
  --trainer.max_epochs=200 \
  --model.loss.init_args.loss_func=models.io.loss.neg_snr \
  --trainer.limit_val_batches=0  # Disable validation

#!/bin/bash
