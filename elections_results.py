import numpy as np
import pandas as pd
import itertools
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
# models that are being considered
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


"""
def grid_search_cross_validation(x_train, y_train, grid, model):
    gridCV = GridSearchCV(model, grid, cv=10)
    gridCV.fit(x_train, y_train.T.squeeze())
    return gridCV.best_params_


def get_svc_best_params(x_train, y_train):
    kernel = ['poly', 'sigmoid']
    degree = [3, 4, 5]
    tol = [ 10**(-3)]
    grid = {
        'kernel' : kernel,
        'degree' : degree,
    }
    res = grid_search_cross_validation(x_train, y_train, grid, SVC())
    print(res)
"""

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



def cross_validation_acc_score(x, y, clf):
    skfold = StratifiedKFold(n_splits=10).split(x, y)
    score = cross_val_score(clf, x, y, cv=skfold)
    print('Accuracy {}%'.format(score.mean()*100))

"""
def find_model():
    label='Vote'
    x_train = pd.read_csv("x_train.csv", header=0)
    y_train = pd.read_csv("y_train.csv", squeeze=True, header=None)
    x_valid = pd.read_csv("x_valid.csv", header=0)
    y_valid = pd.read_csv("y_valid.csv", squeeze=True, header=None)
    x_test = pd.read_csv("x_test.csv", header=0)
    y_test = pd.read_csv("y_test.csv", squeeze=True, header=None)
    #get_random_forest_best_params(x_train, y_train)
    x = x_train
    y = y_train

    # Best parameters for Random Tree Forest: {'criterion': 'gini', 'max_depth': 30, 'max_features': 'auto', 'min_samples_split': 2, 'n_estimators': 50}
    rand_forest_clf = RandomForestClassifier(criterion='gini', max_depth=50, min_samples_split=5, n_estimators=50)
    cross_validation_acc_score(x, y, rand_forest_clf)
    rand_forest_clf.fit(x, y)
    prediction_rand_forest = rand_forest_clf.predict(x_valid)

    # Best parameters for SVC {'degree': 4, 'kernel': 'poly'}
    svm_poly_clf = SVC(kernel='poly', degree=4, probability=True)
    svm_poly_clf.fit(x, y)
    prediction_svm_poly = svm_poly_clf.predict(x_valid)

    # Multi-layer perceptron classifier
    perceptron_clf = MLPClassifier(activation="relu", alpha=0.1, hidden_layer_sizes=(10, 10, 10),
                                   learning_rate="constant", max_iter=2000)
    perceptron_clf.fit(x, y)
    prediction_perceptron = perceptron_clf.predict(x_valid)


    estimators = [
        ('Random Forest',  RandomForestClassifier(criterion='gini', max_depth=50, min_samples_split=5, n_estimators=50)),
        ('SVC', SVC(kernel='poly', degree=4, probability=True)),
        ('Percepton', MLPClassifier(activation="relu", alpha=0.1, hidden_layer_sizes=(10, 10, 10),
                                   learning_rate="constant", max_iter=2000))
    ]
    blend_clf = StackingClassifier(estimators)
    blend_clf.fit(x, y)
    prediction_blend = blend_clf.predict(x_valid)
    # evaluate and plot confusion matrices

    performance_data = [('Random Forest', prediction_rand_forest, y_valid),
                        ('SVM Polinomial Kernel', prediction_svm_poly, y_valid),
                        ('Perceptron', prediction_perceptron, y_valid),
                        ('Blending ', prediction_blend, y_valid)
                        ]
    print_accuracy_scores(performance_data)
    print_f1_score(performance_data)

    prediction = prediction_blend
    parties = np.unique(prediction)

    num_votes_for_party = lambda party: len([vote for vote in prediction if vote == party])
    list_of_parties = [(party, num_votes_for_party(party)) for party in parties]
    num_votes = len(y_test.index)

    winner = max(list_of_parties, key=lambda item: item[1])
    print('Party with most probable majority of votes')
    print(winner[0], ':', winner[1], ',', winner[1] * 100 / num_votes, '%')

    # 2. Division of voters between the parties
    print('Amount of votes per party')
    for party_votes in sorted(list_of_parties, key=lambda votes: votes[1], reverse=True):
        print(party_votes[0], ':', party_votes[1], ',', party_votes[1] * 100 / num_votes, '%')
"""

if __name__ == '__main__':
    label = 'Vote'
    x_train = pd.read_csv("x_train.csv", header=0)
    y_train = pd.read_csv("y_train.csv", squeeze=True, header=None)
    x_valid = pd.read_csv("x_valid.csv", header=0)
    y_valid = pd.read_csv("y_valid.csv", squeeze=True, header=None)
    x_test = pd.read_csv("x_test.csv", header=0)
    y_test = pd.read_csv("y_test.csv", squeeze=True, header=None)
    # get_random_forest_best_params(x_train, y_train)
    x = x_train
    y = y_train

    estimators = [
        ('Random Forest',  RandomForestClassifier(criterion='gini', max_depth=50, min_samples_split=5, n_estimators=50)),
        ('SVC', SVC(kernel='poly', degree=4, probability=True)),
        ('Percepton', MLPClassifier(activation="relu", alpha=0.1, hidden_layer_sizes=(10, 10, 10),
                                   learning_rate="constant", max_iter=2000))
    ]
    blend_clf = StackingClassifier(estimators)
    blend_clf.fit(x, y)
    prediction = blend_clf.predict(x_test)
    # evaluate and plot confusion matrices

    parties = np.unique(prediction)

    num_votes_for_party = lambda party: len([vote for vote in prediction if vote == party])
    list_of_parties = [(party, num_votes_for_party(party)) for party in parties]
    num_votes = len(y_test.index)

    winner = max(list_of_parties, key=lambda item: item[1])
    print('Party with most probable majority of votes')
    print(winner[0], ':', winner[1], ',', winner[1] * 100 / num_votes, '%')

    # 2. Division of voters between the parties
    print('Amount of votes per party')
    for party_votes in sorted(list_of_parties, key=lambda votes: votes[1], reverse=True):
        print(party_votes[0], ':', party_votes[1], ',', party_votes[1] * 100 / num_votes, '%')

    performance_data = [('Blending ', prediction, y_test)]
    print_accuracy_scores(performance_data)
    print_f1_score(performance_data)

