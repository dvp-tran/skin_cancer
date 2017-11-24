#!/usr/bin/env python
# -*- coding: utf-8 -*-

from hyperopt import hp
from Hyper_opt_class import HyperoptOptimizer
import numpy as np
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# ----------------------------------------- Input representation matrix ----------------------------------------------#
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# ---------------------------------Dictionary of models and their space4search -------------------------------------- #
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

space4search = {

    'multinomial_naive_bayes': hp.choice('classifier_type', [
        {
            'type': 'multinomoial_naive_bayes',
            'data_input': hp.choice('input_data', ['bow', 'bow_tfidf', 'word2vec_small', 'word2vec_big']),
            'alpha': hp.uniform('alpha_m', 0.0, 2.0)

        }
    ]
                                         ),

    'Bernoulli_naive_bayes': hp.choice('classifier_type', [
        {
            'type': 'multinomoial_naive_bayes',
            'data_input': hp.choice('input_data', ['bow', 'bow_tfidf', 'word2vec_small', 'word2vec_big']),
            'alpha': hp.uniform('alpha_m', 0.0, 2.0)
        }
    ]),
    'LR': hp.choice('classifier_type', [
        {
            'type': 'LR',
            'data_input': hp.choice('input_data', ['bow', 'bow_tfidf', 'word2vec_small', 'word2vec_big']),
            'C': hp.uniform('C_LR', 0, 10.0)
        }

    ]),

    'svm': hp.choice('classifier_type', [
        {
            'type': 'svm',
            'data_input': hp.choice('input_data', ['bow', 'bow_tfidf', 'word2vec_small', 'word2vec_big']),
            'C': hp.uniform('C', 0, 10.0),
            'kernel': hp.choice('kernel', ['linear', 'rbf']),
            'gamma': hp.uniform('gamma', 0, 20.0)
        }

    ]),

    'RF': hp.choice('classifier_type', [
        {
            'type': 'RF',
            'max_depth': hp.choice('max_depth', range(1100)),
            'max_features': hp.choice('max_features', range(1, 200)),
            'data_input': hp.choice('input_data', ['bow', 'bow_tfidf', 'word2vec_small', 'word2vec_big']),
            'n_estimators': hp.choice('n_estimators', range(1, 1000)),
            'criterion': hp.choice('criterion', ["gini", "entropy"]),
        }
    ]),

    'ET': hp.choice('classifier_type', [{
        'type': 'ET',
        'max_depth': hp.choice('max_depth_ET', range(1100)),
        'max_features': hp.choice('max_features_ET', range(1, 200)),
        'data_input': hp.choice('input_data', ['bow', 'bow_  tfidf', 'word2vec_small', 'word2vec_big']),
        'n_estimators': hp.choice('n_estimators_ET', range(1, 1000)),
        'criterion': hp.choice('criterion_ET', ["gini", "entropy"]),
    }]),

    'k-nn': hp.choice('classifier_type', [{
        'type': 'k-nn',
        'data_input': hp.choice('input_data', ['bow', 'bow_tfidf', 'word2vec_small', 'word2vec_big']),
        'n_neighbors': hp.choice('knn_n_neighbors', range(2, 100))
    }]),

    'GB': hp.choice('classifier_type', [
        {
            'type': 'GB',
            'max_depth': hp.choice('max_depth', range(1100)),
            'max_features': hp.choice('max_features', range(1, 200)),
            'data_input': hp.choice('input_data', ['bow', 'bow_tfidf', 'word2vec_small', 'word2vec_big']),
            'n_estimators': hp.choice('n_estimators', range(1, 1000)),
            'learning_rate': hp.uniform('learning_rate', 0, 1.0),
        }
    ])
}

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# ----------------------------------------- Define the general funct------------------------------------------------ #
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def hyper_opt_f(X_, y_, input_matrices, space4search):
    for key in space4search.keys():
        print('optimization of the model: {}'.format(key))
        print(' ')
        optimizer_ = HyperoptOptimizer(X_, y_, input_matrices, space4search[key])
        optimizer_.fit()
        print(' ')
        print(' ')


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# ----------------------------------------- Launch HyperoptOptimizer ------------------------------------------------ #
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#for key in space4search.keys():
 #   print('optimization of the model: {}'.format(key))
  #  print(' ')
   # X,y = 3, 3
    #optimizer = HyperoptOptimizer(X, y, input_matrices, space4search[key])
    #optimizer.fit()
    #print(' ')
    #print(' ')
