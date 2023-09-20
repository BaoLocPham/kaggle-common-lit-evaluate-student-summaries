python main_infer_stage_2.py \
    parameters.n_fold=4 \
    parameters.root_data_dir="./data" \
    parameters.grade_data_dir="./data" \
    parameters.save_model_dir="./outputs" \
    parameters.debug=True \
    parameters.inference_stage_2.input_dir="./outputs" \
    parameters.inference_stage_2.input_file="stage_1_output.csv" \
    parameters.inference_stage_2.input_model_dir="./outputs" \
    parameters.inference_stage_2.output_dir="./outputs" \
    parameters.inference_stage_2.have_stage_1=True
