from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import scipy.stats


def __plot_parameter_score(parameters, scores, parameter_name, title, metrics_name):
    # assert parameters.shape == scores.shape
    parameters, scores = tuple([np.array(parameters), np.array(scores)])

    plt.clf()
    plt.title(metrics_name)
    plt.xlabel(parameter_name)
    plt.ylabel('CV weighted score')
    if np.issubdtype(parameters.dtype, np.number):
        parameters, scores = tuple([parameters[np.argsort(parameters)], scores[np.argsort(parameters)]])
        x = parameters
    else:
        x = np.arange(len(scores))
        plt.xticks(x, parameters)

    plt.plot(x, scores, 'go--', linewidth=2)
    plt.savefig('plots/{}.png'.format(title))
    #plt.show()


def plot_parameter_score(classifier_name: str, cv_result: dict, scoring: list):
    parameter_to_value_to_best_score = defaultdict(lambda: defaultdict(lambda: float("-inf")))

    for metrics_name in scoring:
        for score, parameter_to_value in zip(cv_result['mean_test_' + metrics_name], cv_result['params']):
            for parameter, value in parameter_to_value.items():
                parameter_to_value_to_best_score[parameter][value] = max(score,
                                                                         parameter_to_value_to_best_score[parameter][value])
        for parameter, value_to_best_score in parameter_to_value_to_best_score.items():
            title = 'Impact of {} on {} {} metrics score'.format(parameter, classifier_name, metrics_name)
            if len(value_to_best_score.keys()) > 1:
                __plot_parameter_score(list(value_to_best_score.keys()),
                                       list(value_to_best_score.values()),
                                       parameter,
                                       title,
                                       metrics_name)


def __plot_parties(x, y, dr, title='Party mapping'):
    plt.clf()
    for party in np.unique(np.array(y)):
        party_mask = np.array(y) == party
        plt.scatter(dr[party_mask, 0], dr[party_mask, 1], marker='.')
    plt.savefig('plots/{}.png'.format(title))
    #plt.show()

def plot_coalition(x, y, coalition, title='Coalition scatter plot'):
    dimension_reducer = PCA(n_components=2).fit_transform(x)
    __plot_parties(x,y, dimension_reducer)
    #coalition_mask = np.array([True if vote in coalition else False for vote in y[0].values])
    coalition_mask = np.isin(np.array(y), coalition)
    #coalition_mask = np.array([item for [item] in coalition_mask])
    #coalition_mask = coalition_mask.reshape(-1, 1)
    plt.clf()
    plt.scatter(dimension_reducer[~coalition_mask, 0],
                dimension_reducer[~coalition_mask, 1],
                marker='.', c='y')
    plt.scatter(dimension_reducer[coalition_mask, 0],
                dimension_reducer[coalition_mask, 1],
                marker='.', c='r')
    #plt.show()
    plt.savefig('plots/{}.png'.format(title))


def plot_gaussians_features(x, y, means, covariance, coalition):
    LABEL='Vote'
    parties = np.unique(y.values)
    gaussian_features = [(index, feat) for index, feat in enumerate(x.columns)
                           if np.unique(x[feat].values).shape[0] > 2]
    index_to_feature = {index: feat for index, feat in gaussian_features}
    for index, feature in gaussian_features:
        plt.figure()
        title = feature + ' distribution'
        plt.title(title)
        feature_mean = means[:, index]
        feature_variance = [covariance[index, index], ] * means.shape[0]
        party_axis = [(party,
                       np.linspace(x.loc[np.array(y[LABEL] == party), feature].min(),
                                   x.loc[np.array(y[LABEL] == party), feature].max(),
                                   num=100)
                       )
                      for party in parties]
        for current_x, mu, sigma in zip(party_axis, feature_mean, feature_variance):
            plt.plot(current_x[1],
                     scipy.stats.norm.pdf(current_x[1], mu, sigma),
                     color='r' if np.isin(current_x[0], coalition) else 'y')
        #plt.show()
        plt.savefig('plots/{}'.format(title))