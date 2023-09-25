python main_train_stage_1.py \
    parameters.n_fold=4 \
    parameters.root_data_dir="./data" \
    parameters.grade_data_dir="" \
    parameters.save_model_dir="./outputs" \
    parameters.debug=True \
    parameters.preprocess_text=False \
    parameters.train_stage_1.select="base" \
    parameters.train_stage_1.full_text=[question,text] \
    parameters.train_stage_1.batch_size=2 \
    parameters.train_stage_1.num_epoch=1 \
    parameters.train_stage_1.max_len=2 \
    parameters.train_stage_1.scheduler="" \
    parameters.train_stage_1.freezing=True \
    parameters.train_stage_1.n_layers_freezing=4 \
    parameters.train_stage_1.strategy="GroupKFold" 