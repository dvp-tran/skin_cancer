
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# ----------------------------------------- Deinfe representation matrix ----------------------------------------------#
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# input_matrices

input_matrices = {
    'in_1': (90, 10),
    'in_2': np.zeros(90, 20)
}

y = 3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# ----------------------------------------- set of training models  ----------------------------------------------#
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
classification_models = {
    "extra trees":  ExtraTreesClassifier(n_estimators=818, max_depth=397, max_features=195),
    "multinomial nb":  MultinomialNB(alpha=0.94),
    "bernoulli nb":  BernoulliNB(),
    "linear svc":  SVC(kernel="rbf", C=5.79),
    "LR":  LogisticRegression(C=9.84),
    "Random forest":  RandomForestClassifier(n_estimators=986, max_depth=242, max_features=183),
    "GB":  GradientBoostingClassifier(learning_rate=0.6103454473476584, n_estimators=37,
                                      max_features=91, max_depth=837),
    "k-nn": KNeighborsClassifier(n_neighbors=3)
}

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# --------------------------------------------Define the score function -----------------------------------------------#
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


scoring = {'F-score': make_scorer(f1_score, {'pos_label': 1, 'average': 'weighted'}),
           'Precision': make_scorer(precision_score, {'pos_label': 1, 'average': 'weighted'}),
           'Recall': make_scorer(recall_score, {'pos_label': 1, 'average': 'weighted'})}

np.random.seed(42)
scores = {}
for name, d in input_matrices.items():
    scores[name] = {}
    for clf, model in classification_models.items():

        try:
            cross_scores = cross_validate(model, d, y, scoring=scoring,
                                          cv=5, return_train_score=False)
            # results = scorer.get_results()
            scores[name][clf] = {}
            for metric_name in cross_scores.keys():
                average_score = np.average(cross_scores[metric_name])
                std_score = np.std(cross_scores[metric_name])
                scores[name][clf][metric_name] = [average_score, std_score]
        except:
            pass


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# -------------------------------------------- Def general function -------------------------------------------------#
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def get_table_score(input_matrices_, labels, models_for_classification):

    np.random.seed(42)
    scores_ = {}
    for name_, d_ in input_matrices_.items():

        scores_[name_] = {}
        for clf_, model_ in models_for_classification.items():

            try:
                cross_scores_ = cross_validate(model_, d, labels, scoring=scoring,
                                               cv=5, return_train_score=False)
                # results = scorer.get_results()
                scores_[name_][clf_] = {}
                for METRIC_NAME in cross_scores_.keys():
                    average_score_ = np.average(cross_scores_[METRIC_NAME])
                    std_score_ = np.std(cross_scores_[METRIC_NAME])
                    scores_[name_][clf_][METRIC_NAME] = [average_score_, std_score_]
            except:
                pass
    l = []
    for matrice_name_, dict_clfs_ in scores_.items():
        temps_2_ = [matrice_name_]
        for clf_name_, dict_scores_ in dict_clfs_.items():
            temps_2_.append(clf_name_)
            for metric_name_, num_scores_ in dict_scores_.items():
                if metric_name_ in ['fit_time', 'score_time']:
                    pass
                else:
                    temps_2_.append(metric_name_)
                    temps_2_.append(num_scores[0])
                    temps_2_.append(num_scores[1])
            l.append(temps_2_)
            temps_2_ = [matrice_name_]
    frame_summary_ = pd.DataFrame(l)
    frame_summary_.columns = ['input_matrix', 'model_name', 'd1', 'mean_f1score', 'std_f1_score',
                              'd2', 'mean_precision', 'std_precision', 'd3', 'mean_recall', 'std_recall']
    frame_summary_ = frame_summary_.drop(['d1', 'd2', 'd3'], axis=1)
    return frame_summary_


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# -------------------------------------------- Create and save the table of scores -----------------------------------#
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == "__main__":
    l = []
    for matrice_name, dict_clfs in scores.items():
        temps_2 = [matrice_name]
        for clf_name, dict_scores in dict_clfs.items():
            temps_2.append(clf_name)
            for metric_name, num_scores in dict_scores.items():
                if metric_name in ['fit_time', 'score_time']:
                    pass
                else:
                    temps_2.append(metric_name)
                    temps_2.append(num_scores[0])
                    temps_2.append(num_scores[1])
            l.append(temps_2)
            temps_2 = [matrice_name]
    frame_summary = pd.DataFrame(l)
    frame_summary.columns = ['input_matrix', 'model_name', 'd1', 'mean_f1score', 'std_f1_score',
                             'd2', 'mean_precision', 'std_precision', 'd3', 'mean_recall', 'std_recall']
    frame_summary = frame_summary.drop(['d1', 'd2', 'd3'], axis=1)
    frame_summary.to_excel('ML_summary_statistic.xlsx')
