#!/bin/bash

# model name, batch size, max seq len
model_configs=(
    "roberta-base 10 32"
    "google/electra-base-discriminator 10 32"
    "microsoft/deberta-base-mnli 6 32"
)

for model_config in "${model_configs[@]}"; do
    IFS=' ' read -ra config <<< "$model_config"
    model="${config[0]}"
    bs="${config[1]}"
    seq_len="${config[2]}"

    eval_bs=$((2 * bs))

    for fold in {0..0}; do
        output="/kaggle/working/${model##*/}_fold${fold}"

        torchrun --nproc_per_node 2 train.py \
            --model_name_or_path "$model" \
            --fold "$fold" \
            --data_dir "./data" \
            --output_dir "$output" \
            --fp16 \
            --num_train_epochs 1 \
            --dataloader_num_workers 4 \
            --learning_rate 4e-5 \
            --weight_decay 0.01 \
            --warmup_ratio 0.1 \
            --optim "adamw_torch" \
            --per_device_train_batch_size "$bs" \
            --per_device_train_batch_size "$eval_bs" \
            --evaluation_strategy "steps" \
            --eval_steps 75 \
            --save_strategy "steps" \
            --save_steps 75 \
            --save_total_limit 1 \
            --report_to "wandb" \
            --metric_for_best_model "mcrmse" \
            --greater_is_better "False" \
            --logging_steps 10 \
            --log_level "error" \
            --disable_tqdm "True" \
            --ddp_find_unused_parameters "False" \
            --dropout 0.0 \
            --add_prompt_question \
            --max_seq_len "$seq_len" \
            --load_best_model_at_end \
            --report_to "wandb" \
            --run_name "test"

#         output_dir="$PWD/$output"
        output_dir="$output"

        # add json files
        for json_file in "$output_dir"/checkpoint*/*token*.json; do
            mv "$json_file" "$output_dir/${json_file##*/}"
        done

        # model files
        for model_file in "$output_dir"/checkpoint*/*model*; do
            mv "$model_file" "$output_dir/${model_file##*/}"
        done

        # remove optimizer states and other files
        to_delete="$(ls -d "$output_dir"/checkpoint* | head -n 1)"
        rm -r "$to_delete"
    done
done
