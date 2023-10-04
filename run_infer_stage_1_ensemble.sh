python3 prepare_ensemble_main_infer_stage_1.py \
    parameters.root_data_dir="./data" \
    parameters.grade_data_dir="" \
    parameters.prepare_ensemble_inference_stage_1.output_dir="./" \
    parameters.prepare_ensemble_inference_stage_1.output_file="test.csv" 

python3 main_infer_stage_1.py \
    parameters.root_data_dir="./data" \
    parameters.grade_data_dir="" \
    parameters.n_fold=4 \
    parameters.inference_stage_1.is_ensemble=True \
    parameters.inference_stage_1.input_prepared_dir="./" \
    parameters.inference_stage_1.input_prepared_file="test.csv" \
    parameters.inference_stage_1.n_fold=1 \
    parameters.inference_stage_1.fold_to_inference=0 \
    parameters.inference_stage_1.max_len=1024 \
    parameters.inference_stage_1.pooling="GemText" \
    parameters.inference_stage_1.model_name="deberta-v3-large" \
    parameters.inference_stage_1.only_model_name="deberta-v3-large" \
    parameters.inference_stage_1.load_model_dir="./weights/version-exalted-microwave/00" \
    parameters.inference_stage_1.have_next_stage=False \
    parameters.inference_stage_1.output_file="submission_0.csv" \
    parameters.inference_stage_1.output_dir="./"

python3 main_infer_stage_1.py \
    parameters.root_data_dir="./data" \
    parameters.grade_data_dir="" \
    parameters.n_fold=4 \
    parameters.inference_stage_1.is_ensemble=True \
    parameters.inference_stage_1.input_prepared_dir="./" \
    parameters.inference_stage_1.input_prepared_file="test.csv" \
    parameters.inference_stage_1.n_fold=1 \
    parameters.inference_stage_1.fold_to_inference=0 \
    parameters.inference_stage_1.max_len=1024 \
    parameters.inference_stage_1.pooling="Mean" \
    parameters.inference_stage_1.model_name="deberta-v3-large" \
    parameters.inference_stage_1.only_model_name="deberta-v3-large" \
    parameters.inference_stage_1.load_model_dir="./weights/version-exalted-microwave/01" \
    parameters.inference_stage_1.have_next_stage=False \
    parameters.inference_stage_1.output_file="submission_1.csv" \
    parameters.inference_stage_1.output_dir="./"

python3 main_infer_stage_1.py \
    parameters.root_data_dir="./data" \
    parameters.grade_data_dir="" \
    parameters.n_fold=4 \
    parameters.inference_stage_1.is_ensemble=True \
    parameters.inference_stage_1.input_prepared_dir="./" \
    parameters.inference_stage_1.input_prepared_file="test.csv" \
    parameters.inference_stage_1.n_fold=1 \
    parameters.inference_stage_1.fold_to_inference=0 \
    parameters.inference_stage_1.max_len=1024 \
    parameters.inference_stage_1.pooling="Max" \
    parameters.inference_stage_1.model_name="deberta-v3-large" \
    parameters.inference_stage_1.only_model_name="deberta-v3-large" \
    parameters.inference_stage_1.load_model_dir="./weights/version-exalted-microwave/02" \
    parameters.inference_stage_1.have_next_stage=False \
    parameters.inference_stage_1.output_file="submission_2.csv" \
    parameters.inference_stage_1.output_dir="./"

python3 main_infer_stage_1.py \
    parameters.root_data_dir="./data" \
    parameters.grade_data_dir="" \
    parameters.n_fold=4 \
    parameters.inference_stage_1.is_ensemble=True \
    parameters.inference_stage_1.input_prepared_dir="./" \
    parameters.inference_stage_1.input_prepared_file="test.csv" \
    parameters.inference_stage_1.n_fold=1 \
    parameters.inference_stage_1.fold_to_inference=0 \
    parameters.inference_stage_1.max_len=1024 \
    parameters.inference_stage_1.pooling="MeanMax" \
    parameters.inference_stage_1.model_name="deberta-v3-large" \
    parameters.inference_stage_1.only_model_name="deberta-v3-large" \
    parameters.inference_stage_1.load_model_dir="./weights/version-exalted-microwave/03" \
    parameters.inference_stage_1.have_next_stage=False \
    parameters.inference_stage_1.output_file="submission_3.csv" \
    parameters.inference_stage_1.output_dir="./"

