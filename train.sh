#!/bin/bash

python SharedTrainer.py fit \
  --config=configs/OnlineSpatialNet.yaml \ # network config
  --config=configs/datasets/sms_wsj_plus.yaml \ # dataset config
  --model.channels=[0,1,2,3,4,5] \ # the channels used
  --model.arch.dim_input=12 \ # input dim per T-F point, i.e. 2 * the number of channels
  --model.arch.dim_output=4 \ # output dim per T-F point, i.e. 2 * the number of sources
  --model.arch.num_freqs=129 \ # the number of frequencies, related to model.stft.n_fft
  --trainer.precision=bf16-mixed \ # mixed precision training, can also be 16-mixed or 32, where 32 can produce the best performance
  --model.compile=true \ # compile the network, requires torch>=2.0. the compiled model is trained much faster
  --data.batch_size=[2,4] \ # batch size for train and val
  --trainer.devices=0 \
  --trainer.max_epochs=100 # better performance may be obtained if more epochs are given
