python3 main_infer_stage_1.py \
    parameters.root_data_dir="./data" \
    parameters.grade_data_dir="./data" \
    parameters.n_fold=4 \
    parameters.debug=True \
    parameters.inference_stage_1.n_fold=4 \
    parameters.inference_stage_1.max_len=2 \
    parameters.inference_stage_1.full_text=[question,text] \
    parameters.inference_stage_1.model_name="microsoft/deberta-v3-base" \
    parameters.inference_stage_1.only_model_name="deberta-v3-base" \
    parameters.inference_stage_1.load_model_dir="./outputs" \
    parameters.inference_stage_1.have_next_stage=False \
    parameters.inference_stage_1.output_file="submission.csv" \
    parameters.inference_stage_1.output_dir="./outputs"