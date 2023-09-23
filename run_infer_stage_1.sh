python3 main_infer_stage_1.py \
    parameters.root_data_dir="./data" \
    parameters.n_fold=4 \
    parameters.inference_stage_1.n_fold=4 \
    parameters.inference_stage_1.fold_to_inference=0 \
    parameters.inference_stage_1.max_len=512 \
    parameters.inference_stage_1.model_name="microsoft/deberta-v3-base" \
    parameters.inference_stage_1.only_model_name="deberta-v3-base" \
    parameters.inference_stage_1.load_model_dir="./weights" \
    parameters.inference_stage_1.have_next_stage=True \
    parameters.inference_stage_1.output_file="stage_1_output.csv" \
    parameters.inference_stage_1.output_dir="./"


python3 main_infer_stage_1.py \
    parameters.root_data_dir="./data" \
    parameters.grade_data_dir="./data" \
    parameters.n_fold=4 \
    parameters.debug=True \
    parameters.inference_stage_1.n_fold=4 \
    parameters.inference_stage_1.max_len=512 \
    parameters.inference_stage_1.max_len_char_title=100 \
    parameters.inference_stage_1.max_len_char_question=100 \
    parameters.inference_stage_1.max_len_char_prompt_text=100\
    parameters.inference_stage_1.full_text=[question,text] \
    parameters.inference_stage_1.model_name="microsoft/deberta-v3-base" \
    parameters.inference_stage_1.only_model_name="deberta-v3-base" \
    parameters.inference_stage_1.load_model_dir="./outputs" \
    parameters.inference_stage_1.have_next_stage=False \
    parameters.inference_stage_1.output_file="stage_1_output.csv" \
    parameters.inference_stage_1.output_dir="./outputs"