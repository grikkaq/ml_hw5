from time import time
import sys
import pandas as pd
import numpy as np
from plotter import *
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import homogeneity_score, \
                            completeness_score, \
                            adjusted_rand_score, \
                            silhouette_score, \
                            davies_bouldin_score, \
                            fowlkes_mallows_score, \
                            v_measure_score
from sklearn.metrics import make_scorer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, \
                                    StratifiedShuffleSplit

LABEL='Vote'

def load_data():
    def __load_data(df_name):
        x = pd.read_csv(df_name + '_set_x.csv', index_col=0)
        y = pd.read_csv(df_name + '_set_y.csv', index_col=0, header=0).rename(index=str)
        return x, y

    x_train, y_train = __load_data('train')
    x_train.reset_index(drop=True, inplace=True), y_train.reset_index(drop=True, inplace=True)
    x_val, y_val = __load_data('valid')
    x_val.reset_index(drop=True, inplace=True), y_val.reset_index(drop=True, inplace=True)
    x_test, y_test = __load_data('test')
    x_test.reset_index(drop=True, inplace=True), y_test.reset_index(drop=True, inplace=True)
    return x_train, y_train, x_val, y_val, x_test, y_test


def __select_params_Gaussian_mixture(x, y, classifier, parameters: dict, scoring=None, refit='adjusted_rand_score'):
    clf = GridSearchCV(
        classifier(),
        parameters,
        cv=StratifiedShuffleSplit(n_splits=3, random_state=42),
        refit=refit,
        scoring=scoring,
        return_train_score=True,
        verbose=sys.maxsize)

    y[LABEL] = pd.factorize(y[LABEL], sort=True)[0]
    clf.fit(x, y.values.ravel())
    plot_parameter_score(classifier.__name__, clf.cv_results_, scoring)


def select_params_Gaussian_mixture():
    x, y, _, _, _, _ = load_data()

    scoring = {'homogeneity_score': make_scorer(homogeneity_score),
               'adjusted_rand_score': make_scorer(adjusted_rand_score),
               #'silhouette_score': make_scorer(silhouette_score),
               'v_measure_score': make_scorer(v_measure_score),
               'completeness_score': make_scorer(completeness_score),
               #'fowlkes_mallows_score': make_scorer(fowlkes_mallows_score)
               }
    # sizes = [1, 5, 10, 20, 40, 60, 80, 100]
    sizes = [2**i for i in range(1,7)]
    sizes = sizes + [80, 100, 128]
    # cov_types = ['full', 'tied']
    # params = {'n_components' : [2**i for i in range(0, 11)], 'covariance_type' : ['full', 'tied', 'diag', 'spherical']}
    params = {'n_components': sizes}
    __select_params_Gaussian_mixture(x, y, classifier=GaussianMixture, parameters=params, scoring=scoring)


def find_steady_coalition():
    x_train, y_train, x_val, y_val, _, _ = load_data()
    x = x_val
    y = y_val

    #gmm = GaussianMixture(n_components=30)
    gmm = GaussianMixture(n_components=40)
    gmm, _ = cross_validation_train(x_train, y_train, gmm)
    clusters_votes = gmm.predict(x)
    clusters_df = pd.DataFrame(data=clusters_votes)

    cluster_size = {}
    cluster_variance = {}

    for cluster in np.unique(clusters_votes):
        cluster_size[cluster] = clusters_df.iloc[clusters_votes == cluster].shape[0]
        cluster_variance[cluster] = np.mean(pd.DataFrame.var(x.iloc[clusters_votes == cluster]))

    clusters_sorted_by_var = sorted(cluster_variance.items(), key=lambda item: item[1])
    print('Clusters sorted by variance: {}'.format(clusters_sorted_by_var))

    parties_in_coalition = []
    threshold = 0.51
    take_party_threshold = 0.03
    num_votes_for_coalition=0
    for cluster, _ in clusters_sorted_by_var:
        parties_in_cluster = np.unique(np.array(y)[clusters_votes == cluster])
        #print('Cluster {}: its parties are \n  {}'.format(cluster, parties_in_cluster))
        for party in parties_in_cluster:
            if party in parties_in_coalition:
                continue
            votes_ratio = np.sum((clusters_df[0] == cluster) & (y[LABEL] == party)) / \
                                   np.sum(y[LABEL] == party)
            if votes_ratio < take_party_threshold:
                continue
            print('From cluster {} taking party {} with {} votes out a {} for cluster and {} for party.'.format(
                cluster, party,
                np.sum((clusters_df[0] == cluster) & (y[LABEL] == party)),
                np.sum(clusters_df[0] == cluster),
                np.sum(y[LABEL] == party)
            ))
            parties_in_coalition.append(party)
            num_votes_for_coalition += np.sum(y[LABEL] == party)
            if num_votes_for_coalition >  threshold * len(clusters_votes):
                break
        if num_votes_for_coalition > threshold * len(clusters_votes):
            break

    print('Coalition: {}'.format(parties_in_coalition))
    print('{} coalition votes vs {} '.format(num_votes_for_coalition,
                                             len(clusters_votes) - num_votes_for_coalition))
    plot_coalition(x, y, parties_in_coalition, title='Coalition clustering validation')



if __name__ == '__main__':
    #select_params_Gaussian_mixture()
    find_steady_coalition()
    #stedy_coaition_analysis()
