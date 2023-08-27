class CFG:
    
    # path, model type, max seq len, batch size, add prompt question
    model_configs = [
        ("/content/train_output/roberta-base_fold{fold}", "roberta", 512, 48, True),
    ]
    # if None, use average of preds 
    content_weights = None
    wording_weights = None
    
    do_eval = False # set to True if you want to do validation
    do_predict = True