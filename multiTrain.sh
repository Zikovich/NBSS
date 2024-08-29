#!/bin/bash

# Define the list of loss functions to iterate over
loss_functions=(
  "models.io.loss.neg_si_sdr"
  "models.io.loss.neg_sa_sdr"
  #"models.io.loss.neg_snr"
  "models.io.loss.cirm_mse"
  "models.io.loss.cc_mse"
)

# Iterate over each loss function
for loss_func in "${loss_functions[@]}"; do
  echo "Running training with loss function: $loss_func"
  
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
    --model.loss.init_args.loss_func=$loss_func \
    --trainer.limit_val_batches=0  # Disable validation
  
  echo "Finished training with loss function: $loss_func"
done
