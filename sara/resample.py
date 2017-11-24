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


def random_undersampling(X, Y, n_sample=5):
    """
    X,y : numpy arrays
    return :
    5 random ensemble of indices general_balenced_set:
        general_balenced_set[0] = the shuffeled indices that inssure the class balance
    """
    indices = np.array(range(len(Y)))
    positive_samples = indices[Y==1]
    negative_samples = indices[Y==0]
    general_balenced_set = []
    for k in range(n_sample):
        indices_ = np.random.choice(negative_samples, 1483, replace=False)
        # append positive and negative
        balenced_set = np.append(indices_, positive_samples)
        # shuffle indices
        balenced_set = np.random.shuffle(balenced_set)
        general_balenced_set.append(balenced_set)
    return general_balenced_set