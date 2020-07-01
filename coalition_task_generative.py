import sys
import pandas as pd
import numpy as np
from plot_tools import *
from sklearn.metrics import homogeneity_score, \
                            completeness_score, \
                            adjusted_rand_score, \
                            silhouette_score, \
                            davies_bouldin_score, \
                            fowlkes_mallows_score, \
                            v_measure_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from itertools import chain, combinations
LABEL='Vote'



def get_possible_coalitions(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) // 2 + 1))


def load_data():
    def __load_data(df_name):
        x = pd.read_csv('x_' + df_name + '.csv', header=0)
        y = pd.read_csv('y_' + df_name + '.csv', squeeze=True, header=None).rename(index=str)
        return x, y

    x_train, y_train = __load_data('train')
    x_val, y_val = __load_data('valid')
    x_test, y_test = __load_data('test')
    return x_train, y_train, x_val, y_val, x_test, y_test


def cross_validation_acc_score(x, y, clf):
    skfold = StratifiedKFold(n_splits=10).split(x, y)
    score = cross_val_score(clf, x, y, cv=skfold)
    print(score)
    print('Accuracy {}%'.format(score.mean()*100))
    return clf


def grid_search_cross_validation(x_train, y_train, grid, model):
    gridCV = GridSearchCV(model, grid, cv=10)
    gridCV.fit(x_train, y_train.T.squeeze())
    return gridCV.best_params_


def get_lda_best_params():
    x_train, y_train, _, _, _, _ = load_data()
    solver = [  'lsqr', 'eigen']
    shrinkage = [ 'auto']
    grid = {
        'solver' : solver,
        'shrinkage' : shrinkage
    }
    res = grid_search_cross_validation(x_train, y_train, grid, LinearDiscriminantAnalysis())
    print(res)

def print_accuracy_scores(performance_data):
    print('Accuracy scores:')
    for i, data in enumerate(performance_data):
        model_name = data[0]
        pred = data[1]
        test = data[2]
        acc = metrics.accuracy_score(y_true=pred, y_pred=test, normalize=True)
        print(model_name + ' accuracy: ', acc)

def print_f1_score(performance_data):
    print('f1 scores:')
    for i, data in enumerate(performance_data):
        model_name = data[0]
        pred = data[1]
        test = data[2]
        acc = metrics.f1_score(y_true=pred, y_pred=test, average='macro')
        print(model_name + ' f1 score: ', acc)


def find_model():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    # all_df = pd.concat([train_df, validation_df, test_df])

    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', store_covariance=False)
    qda = QuadraticDiscriminantAnalysis()
    random_forest = RandomForestClassifier(criterion='gini', max_depth=50, min_samples_split=5, n_estimators=50)

    print('Linear Discriminant Analysis: ', end='')
    cross_validation_acc_score(x_train, y_train, lda)
    print('Random Forest: ', end='')
    cross_validation_acc_score(x_train, y_train, random_forest)

    lda.fit(x_train, y_train), random_forest.fit(x_train, y_train)

    prediction_lda = lda.predict(x_val)
    prediction_random_forest = random_forest.predict(x_val)

    performance_data = [
        ('Linear Discriminant Analysis', prediction_lda, y_val),
        ('Random Forest', prediction_random_forest, y_val)
    ]
    print_accuracy_scores(performance_data)
    print_accuracy_scores(performance_data)


def find_steady_coalition():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    # trying to implement LDA with Least Squares solver
    #clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', store_covariance=True)
    clf = RandomForestClassifier(criterion='gini', max_depth=50, min_samples_split=5, n_estimators=50)
    clf.fit(x_train, y_train)
    parties_list = np.unique(y_train.values)
    feature_to_index_map = {clf.classes_[i]: i for i in range(len(clf.classes_))}
    # prediction
    probabilities_per_voter = clf.predict_proba(x_val)

    best_coalition = []
    best_coalition_v_score = float(-np.inf)
    best_coalition_homo = float(-np.inf)

    for possible_coalition in get_possible_coalitions(parties_list):
        y_coalition = np.isin(y_val.values.ravel(), possible_coalition)
        probabilities_coalition = np.sum(probabilities_per_voter[:,
                                         [feature_to_index_map[feat] for feat in possible_coalition]],
                                         axis=1)
        coalition_score = np.mean(probabilities_coalition)

        if (coalition_score < 0.51):
            continue

        voters_likely_to_vote = [ voter > 0.5 for voter in probabilities_coalition ]
        standart_deviation = np.std(probabilities_coalition)
        val_predict_score = np.mean(voters_likely_to_vote)
        v_score = v_measure_score(y_coalition, voters_likely_to_vote)
        homo_score = homogeneity_score(y_coalition, voters_likely_to_vote)
        #print('Homogeneity score: {} \nV-Measure score: {} '.format(homo_score, v_score))
        #print('Predicition mean {} and std {}'.format(val_predict_score, standart_deviation))

        #if v_score > best_coalition_v_score:
        if v_score > best_coalition_v_score:
            best_coalition = possible_coalition
            best_coalition_v_score = homo_score

    plot_coalition(x_train, y_train, best_coalition)
    print(best_coalition)
    print('Coalition: {}'.format(best_coalition))
    print('{} coalition votes vs {} '.format(np.sum(voters_likely_to_vote),
                                             len(y_val) - np.sum(voters_likely_to_vote)))

    #lda_tst = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', store_covariance=True)
    random_forest = RandomForestClassifier(criterion='gini', max_depth=50, min_samples_split=5, n_estimators=50)
    print('Test: ', end='')
    random_forest.fit(x_train, y_train)

    prediction_random_forest = random_forest.predict(x_val)

    performance_data = [('Random Forest', prediction_random_forest, y_val)]
    print_accuracy_scores(performance_data)
    print_accuracy_scores(performance_data)
    #plot_coalition(x_val, y_val, best_coalition, title='Coalition [LDA]')



if __name__ == '__main__':
    #select_params_Gaussian_mixture()
    #get_lda_best_params()
    #find_model()
    find_steady_coalition()