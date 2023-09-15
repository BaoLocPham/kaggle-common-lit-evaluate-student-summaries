python main_infer_stage_2.py \
    parameters.n_fold=4 \
    parameters.root_data_dir="./data" \
    parameters.save_model_dir="./outputs" \
    parameters.debug=True \
    parameters.infer_stage_2.input_model_dir="./outputs" \
    parameters.infer_stage_2.input_model_name="lgbm_{target}_{fold}.pkl"
