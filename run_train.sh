python main_train.py \
    parameters.n_fold=4 \
    parameters.root_data_dir="./data" \
    parameters.save_model_dir="./outputs" \
    parameters.debug=True \
    parameters.preprocess_text=True \
    parameters.model.select="base" \
    parameters.model.batch_size=2 \
    parameters.model.num_epoch=5 \
    parameters.model.max_len=1024 \
    parameters.model.freezing=True \
    parameters.model.n_layers_freezing=4 \
    parameters.model.strategy="GroupKFold" 