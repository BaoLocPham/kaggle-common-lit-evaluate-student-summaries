python main_train_stage_2.py \
    parameters.n_fold=4 \
    parameters.root_data_dir="./data" \
    parameters.save_model_dir="./outputs" \
    parameters.debug=True \
    parameters.train_stage_2.output_model_dir="./outputs" 

python main_train_stage_2.py \
    parameters.root_data_dir="./data" \
    parameters.n_fold=4 \
    parameters.debug=True \
    parameters.train_stage_2.output_model_dir="./outputs" \
    parameters.train_stage_2.strategy="GroupKFold_grade" 