#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1

python liteSharedTrainer.py fit \
  --config=configs/liteOnlineSpatialNet.yaml \
  --config=configs/datasets/clarity_data_config.yaml \
  --model.channels="[0,1,2,3,4,5]" \
  --model.arch.dim_input=12 \
  --model.arch.dim_output=2 \
  --model.arch.num_freqs=129 \
  --trainer.precision=32-true \
  --model.compile=false \
  --data.batch_size=[4,1] \
  --trainer.devices="1" \
  --data.init_args.num_workers=0 \
  --data.init_args.persistent_workers="False" \
  --trainer.max_epochs=100 \
  --model.loss.init_args.loss_func=models.io.loss.neg_snr \
  --trainer.accumulate_grad_batches=4 \
  --data.init_args.train_limit=0 \
  --data.init_args.val_limit=0 \
  --data.init_args.test_limit=0 \
  

#!/bin/bash
