python main_infer.py \
    parameters.root_data_dir="./data" \
    parameters.n_fold=4 \
    parameters.inference.n_fold=4 \
    parameters.inference.fold_to_inference=0 \
    parameters.inference.max_len=512 \
    parameters.inference.model_name="microsoft/deberta-v3-base" \
    parameters.inference.only_model_name="deberta-v3-base" \
    parameters.inference.load_model_dir="./weights" \
    parameters.inference.output_dir="./"