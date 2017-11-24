import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE


def resampling_features_matrix(X, Y):
    sm = SMOTE(random_state=42, ratio='minority')
    X_res, Y_res = sm.fit_sample(X, Y)
    return X_res, Y_res


def evaluate_imbalanced_ML(model, X, Y, n_split):
    scores = []
    kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)
    sm = SMOTE(random_state=42, ratio='minority')
    i = 0
    for train, test in kfold.split(X, Y):
        x_res, y_res = sm.fit_sample(X[train], Y[train])
        print(np.bincount(Y[train]))
        print(np.bincount(y_res))
        print('training model for fold %d on related oversample minority classes' % i)
        model.fit(x_res, y_res)
        i += 1
        print('Compute test average-precision for fold %d' % i)
        preds = model.predict(X[test])
        precision, recall, pr_tresh = precision_recall_curve(Y[test], preds)
        scores.append([precision.mean(), recall.mean()])
        print('the precision is %d' % precision.mean())
    return scores
