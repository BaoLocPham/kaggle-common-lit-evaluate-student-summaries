python main_train_stage_1.py \
    parameters.n_fold=4 \
    parameters.root_data_dir="./data" \
    parameters.save_model_dir="./outputs" \
    parameters.debug=True \
    parameters.preprocess_text=True \
    parameters.model.select="base" \
    parameters.model.batch_size=2 \
    parameters.model.num_epoch=5 \
    parameters.model.max_len=512 \
    parameters.model.freezing=True \
    parameters.model.n_layers_freezing=4 \
    parameters.model.strategy="GroupKFold" 


python main_train_stage_1.py \
    parameters.n_fold=4 \
    parameters.root_data_dir="./data" \
    parameters.grade_data_dir="" \
    parameters.save_model_dir="./outputs" \
    parameters.debug=True \
    parameters.preprocess_text=False \
    parameters.train_stage_1.select="base" \
    parameters.train_stage_1.batch_size=2 \
    parameters.train_stage_1.num_epoch=1 \
    parameters.train_stage_1.max_len=2 \
    parameters.train_stage_1.max_len_char_title=100 \
    parameters.train_stage_1.max_len_char_question=100 \
    parameters.train_stage_1.scheduler="" \
    parameters.train_stage_1.freezing=True \
    parameters.train_stage_1.n_layers_freezing=4 \
    parameters.train_stage_1.strategy="GroupKFold" 