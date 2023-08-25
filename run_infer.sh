#!/bin/bash

# Define model configurations
declare -a model_configs=(
    "/kaggle/working/roberta-base_fold{fold} roberta 32 48 true"
    "./electra-base-discriminator_fold{fold} electra 32 48 true"
    "./deberta-base-mnli_fold{fold} deb-v1 32 32 true"
)

do_eval=true
do_predict=true

suffix="_eval"

for model_config in "${model_configs[@]}"; do
    IFS=" " read -ra config <<< "$model_config"
    model_path="${config[0]}"
    model_type="${config[1]}"
    max_seq_len="${config[2]}"
    batch_size="${config[3]}"
    add_prompt_q="${config[4]}"

    output_dir="$model_path$suffix"
    tokenized_ds_path="${model_type}_tokenized$suffix"

    torchrun --nproc_per_node 2 infer.py \
      --model_name_or_path $model_path \
      --folds "0" \
      --data_dir "./data" \
      --output_dir  $output_dir \
      --dataloader_num_workers 2 \
      --per_device_eval_batch_size $batch_size \
      --save_strategy "no" \
      --report_to "none" \
      --log_level "warning" \
      --tokenized_ds_path $tokenized_ds_path \
      --do_eval \
      --disable_tqdm True \
      --max_seq_length $max_seq_len \
      --add_prompt_question $add_prompt_q
done