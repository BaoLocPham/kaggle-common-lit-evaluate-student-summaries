import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_mcrmse(eval_pred):
    """
    Calculates mean columnwise root mean squared error
    https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview/evaluation
    """
    preds, labels = eval_pred

    col_rmse = np.sqrt(np.mean((preds - labels) ** 2, axis=0))
    mcrmse = np.mean(col_rmse)

    return {
        "content_rmse": col_rmse[0],
        "wording_rmse": col_rmse[1],
        "mcrmse": mcrmse,
    }

def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]#.detach().to('cpu').numpy()
        y_pred = y_preds[:,i]#.detach().to('cpu').numpy()
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores

def score_loss(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return {
        'mcrmse_score' : mcrmse_score,
        'content_score' : scores[0],
        'wording_score' : scores[1]
    }