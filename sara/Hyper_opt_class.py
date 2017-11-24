
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, STATUS_OK, Trials
from sklearn.ensemble import ExtraTreesClassifier


class HyperoptOptimizer(object):
    def __init__(self, input_mat, y, set_matrices,  space4search, scoring='f1_weighted'):
        """
        :param space4search:
        :param scoring:
        :return:
        """
        self.input_mat = input_mat
        self.space4search = space4search
        self.scoring = scoring
        self.best = 0
        self.y = y
        self.set_matrices = set_matrices

    def hyperopt_train_test(self, params):

        t = params['type']
        dtype = params['data_input']

        del params['type']
        del params['data_input']

        try:
            self.input_mat = self.set_matrices[dtype].toarray()
        except:
            self.input_mat = self.set_matrices[dtype]

        if t == 'multinomial_naive_bayes':
            from sklearn.naive_bayes import MultinomialNB
            clf = MultinomialNB(**params)

        elif t == 'Bernoulli_naive_bayes':
            from sklearn.naive_bayes import BernoulliNB
            clf = BernoulliNB(**params)

        elif t == 'LR':
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(**params)

        elif t == 'svm':
            from sklearn.svm import SVC
            clf = SVC(**params)

        elif t == 'ET':
            clf = ExtraTreesClassifier(**params)

        elif t == 'RF':
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(**params)

        elif t == 'k-nn':
            from sklearn.neighbors import KNeighborsClassifier
            clf = KNeighborsClassifier(**params)
        elif t == 'GB':
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier()

        else:
            return 0

        return cross_val_score(clf, self.input_mat, self.y, scoring=self.scoring, cv=5).mean()

    def f(self, params):
        acc = self.hyperopt_train_test(params)
        if acc > self.best:
            self.best = acc
            print('new best:',  self.best,  params)
        return {'loss': -acc, 'status': STATUS_OK}

    def fit(self):
        trials = Trials()
        best = fmin(self.f, self.space4search, algo=tpe.suggest, max_evals=20, trials=trials)
        print('best:')
        print(best)
