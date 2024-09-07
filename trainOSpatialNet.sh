#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1

python SharedTrainer.py fit \
  --config=configs/onlineSpatialNet.yaml \
  --config=configs/datasets/clarity_data_config.yaml \
  --model.channels="[0,1,2,3,4,5]" \
  --model.arch.dim_input=12 \
  --model.arch.dim_output=2 \
  --model.arch.num_freqs=129 \
  --model.optimizer="[AdamW, {lr: 0.001, weight_decay: 0.001}]" \
  --trainer.precision=16-mixed \
  --model.compile=true \
  --data.batch_size=[1,1] \
  --trainer.devices="4" \
  --data.init_args.num_workers=11 \
  --data.init_args.persistent_workers="True" \
  --trainer.max_epochs=100 \
  --model.loss.init_args.loss_func=models.io.loss.neg_snr \
  --trainer.accumulate_grad_batches=4 \
  --trainer.strategy="ddp" \
  --ckpt_path="/teamspace/studios/this_studio/NBSS/logs/OnlineSpatialNet/version_68/checkpoints/last.ckpt"



#!/bin/bash
