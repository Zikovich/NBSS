#!/bin/bash

# Define the list of loss functions to iterate over
loss_functions=(
  #"models.io.loss.neg_si_sdr"
  #"models.io.loss.neg_sa_sdr"
  "models.io.loss.neg_snr"
  #"models.io.loss.cirm_mse"
  #"models.io.loss.cc_mse"
  #"models.io.loss.neg_nb_pesq"  # NB PESQ loss function with internal resampling
  #"models.io.loss.neg_wb_pesq"  # WB PESQ loss function with internal resampling
  #"NegSTOILoss_ext"
  #"combined_stoi_si_sdr_snr_25_25_50"
  #"combined_stoi_si_sdr_snr_25_0_75" 
  #"combined_stoi_si_sdr_snr_25_75_0" 
  #"combined_stoi_si_sdr_snr_15_10_75"      # STOI loss function (extended)
)

# Iterate over each loss function
for loss_func in "${loss_functions[@]}"; do
  echo "Running training with loss function: $loss_func"

  # Set the appropriate loss function and arguments
  if [ "$loss_func" == "models.io.loss.neg_nb_pesq" ]; then
    extra_args="--model.loss.init_args.loss_func_kwargs.sample_rate=8000"
  elif [ "$loss_func" == "models.io.loss.neg_wb_pesq" ]; then
    extra_args="--model.loss.init_args.loss_func_kwargs.sample_rate=16000"
  elif [ "$loss_func" == "NegSTOILoss_ext" ]; then
    extra_args="--model.loss.init_args.loss_func=models.io.loss.neg_stoi_loss \
                --model.loss.init_args.loss_func_kwargs.sample_rate=48000 \
                --model.loss.init_args.loss_func_kwargs.extended=true \
                --model.loss.init_args.loss_func_kwargs.use_vad=false \
                --model.loss.init_args.loss_func_kwargs.do_resample=false"
  elif [ "$loss_func" == "NegSTOILoss_no_ext" ]; then
    extra_args="--model.loss.init_args.loss_func=torch_stoi.NegSTOILoss \
                --model.loss.init_args.loss_func_kwargs.sample_rate=48000 \
                --model.loss.init_args.loss_func_kwargs.extended=false \
                --model.loss.init_args.loss_func_kwargs.use_vad=false \
                --model.loss.init_args.loss_func_kwargs.do_resample=true"
  elif [ "$loss_func" == "combined_stoi_si_sdr_snr_25_25_50" ]; then
    extra_args="--model.loss.init_args.loss_func=models.io.loss.combined_stoi_si_sdr_snr \
                --model.loss.init_args.loss_func_kwargs.stoi_weight=0.25 \
                --model.loss.init_args.loss_func_kwargs.si_sdr_weight=0.25 \
                --model.loss.init_args.loss_func_kwargs.snr_weight=0.50"
  elif [ "$loss_func" == "combined_stoi_si_sdr_snr_25_0_75" ]; then
    extra_args="--model.loss.init_args.loss_func=models.io.loss.combined_stoi_si_sdr_snr \
                --model.loss.init_args.loss_func_kwargs.stoi_weight=0.25 \
                --model.loss.init_args.loss_func_kwargs.si_sdr_weight=0 \
                --model.loss.init_args.loss_func_kwargs.snr_weight=0.75"
  elif [ "$loss_func" == "combined_stoi_si_sdr_snr_25_75_0" ]; then
    extra_args="--model.loss.init_args.loss_func=models.io.loss.combined_stoi_si_sdr_snr \
                --model.loss.init_args.loss_func_kwargs.stoi_weight=0.25 \
                --model.loss.init_args.loss_func_kwargs.si_sdr_weight=0.75 \
                --model.loss.init_args.loss_func_kwargs.snr_weight=0"
  elif [ "$loss_func" == "combined_stoi_si_sdr_snr_15_10_75" ]; then
    extra_args="--model.loss.init_args.loss_func=models.io.loss.combined_stoi_si_sdr_snr \
                --model.loss.init_args.loss_func_kwargs.stoi_weight=0.15 \
                --model.loss.init_args.loss_func_kwargs.si_sdr_weight=0.10 \
                --model.loss.init_args.loss_func_kwargs.snr_weight=0.75"
  else
    extra_args=""
  fi

  # Run the training with the appropriate loss function and arguments
  python SharedTrainer.py fit \
    --config=configs/onlineSpatialNet.yaml \
    --config=configs/datasets/clarity_data_config.yaml \
    --model.loss.init_args.loss_func=$loss_func \
    --model.channels="[0,1,2,3,4,5]" \
    --model.arch.dim_input=12 \
    --model.arch.dim_output=2 \
    --model.arch.num_freqs=129 \
    --trainer.precision=16-mixed \
    --model.compile=true \
    --data.batch_size="1" \
    --trainer.devices=1 \
    --trainer.max_epochs=200 \
    --data.init_args.train_limit=1 \
    --data.init_args.val_limit=1 \
    --data.init_args.test_limit=1 \
    $extra_args

  echo "Finished training with loss function: $loss_func"
done
