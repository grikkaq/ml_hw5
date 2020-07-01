#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from outliers import smirnov_grubbs as grubbs
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, SelectFromModel, mutual_info_classif
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import LocalOutlierFactor


def set_types(data):
    # Set the types for all the features and split to features and labels.
    object_features = data.keys()[data.dtypes.map(lambda x: x == 'object')]
    for f in object_features:
        data[f] = data[f].astype("category")


def split_dataset(X, y):
    # Splitting into train/valid/test in proportions 0.6/0.2/0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, stratify=y_train)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def categorical_to_numerical(df, y):
    # Transform feature of two possible values to boolean feature.
    # Other (categorical) features except label are transformed into one-hot representation.
    df = df.reset_index(drop=True)
    categorical_features = list(df.select_dtypes(include='category').keys())

    le = LabelEncoder()
    y = le.fit_transform(y)
    for f in categorical_features:
        unique_val = np.unique(df[f])
        if len(unique_val) == 2:
            df[f] = df[f].apply(lambda item: int(item == unique_val[0]))
        else:
            column_dummies = pd.get_dummies(df[f])
            column_dummies.rename(columns={dummy: f + '-is-' + dummy for dummy in column_dummies.keys()},
                                  inplace=True)
            df = pd.concat([df.drop(f, axis=1), column_dummies],
                           axis=1)
    return df


'''     Missed Values       '''
def closet_fit(df):
    '''
    Filling missed values.
    Categorical features are filled with the same value as the closet
    sample from data set. Numerical features filled in following way:
    pick pool of 10 closet samples and then fill with average value
    of pool.
    '''
    def k_closest_samples(row, data, k, categorical_features, numerical_features):
        data.reset_index(drop=True)
        cat_diffs = (np.array(data[categorical_features]) != np.array(row[categorical_features]))
        num_diffs = np.abs(data[numerical_features] - np.squeeze(np.array(row[numerical_features])))
        num_diffs /= np.max(data[numerical_features])
        diffs = np.array(cat_diffs.sum(axis=1)) + np.array(num_diffs.sum(axis=1))
        closest_rows = np.argpartition(diffs, k)[:k]
        return data.iloc[closest_rows]

    categorical_features = df.select_dtypes(include='category').keys()
    numerical_features = df.select_dtypes(exclude='category').keys()
    no_nan_df = df.dropna(axis=0, inplace=False).reset_index(drop=True)
    for index, row in df.iterrows():
        if not row.isna().any():
            continue
        #print(index)
        for f in df.columns[row.isna()]:
            if f in categorical_features:
                row.at[f] = k_closest_samples(row=row, data=no_nan_df, k=1,
                                              categorical_features=categorical_features.drop(f),
                                              numerical_features=numerical_features).iloc[0][f]
            else:
                ten_closet_df = k_closest_samples(row=row, data=no_nan_df, k=100,
                                                  categorical_features=categorical_features,
                                                  numerical_features=numerical_features.drop(f))
                row.at[f] = ten_closet_df[f].mean()
        df.loc[index] = row
        no_nan_df.append(row.rename(), ignore_index=True)
    return df


'''     Noise Outlier   '''
def noise_outlier_variance(df, k):
    # Drop samples which values exceed k multiplied Standard Derivation
    numerical_features = df.select_dtypes(exclude='category').keys()
    df = df[df[numerical_features].apply(lambda x: (x - x.mean()).abs() < (k * x.std())).all(1)]
    df = df.dropna(axis=0)
    return df

def remove_noise_outliers(df, y):
    outlier_clf = LocalOutlierFactor(n_neighbors=20)
    mask = [True if item == 1 else False for item in outlier_clf.fit_predict(df)]
    return df.iloc[mask, :]


'''    Scaling      '''
def scale_data(df: pd.DataFrame):
    df.reset_index(drop=True, inplace=True)
    normalized_data = df.copy()
    for feature in df.select_dtypes(exclude='category').keys():
        column = np.array(df[feature])
        if kurtosis(column) < 0 or len(feature.split('-')) != 1:
            #linear scaling to (-1,1)
            if column.max() == column.min():
                scaled_feature = column / column.max() if column.max() != 0 else column
            else:
                column_std = (column - column.min()) / (column.max() - column.min())
                scaled_feature = column_std * (1 - (-1))  + (-1)
        else:
            mean = column.mean()
            std = column.std()
            scaled_feature = (column - mean) / std
        normalized_data[feature] = scaled_feature
    return normalized_data

def scale_MinMax(df):
    df = df.reset_index(drop=True)
    numerical_features = df.select_dtypes(exclude='category').keys()
    categorical_features = df.select_dtypes(include='category').keys()
    scaler = MinMaxScaler()
    numerical_scaled_df = pd.DataFrame(data=scaler.fit_transform(df[numerical_features]),
                                        columns=numerical_features)
    return pd.concat([numerical_scaled_df, df[categorical_features]], axis=1)


'''     Feature Selection       '''
def extract_features(df, mask):
    return df.columns[mask]

def feature_select_filter(X, y, k):
    # select k=25 best features
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit_transform(X.values, y.values)
    return extract_features(X, mask=selector.get_support())

def feature_select_SVM_wrapper(X, y):
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=5000)
    selector = SelectFromModel(lsvc, prefit=False)
    selector.fit_transform(X.values, y.values)
    return extract_features(X, mask=selector.get_support())

def feature_select_DT_wrapper(X, y):
    clf = ExtraTreesClassifier(n_estimators=100)
    selector = SelectFromModel(clf, prefit=False)
    selector.fit_transform(X.values, y.values)
    return extract_features(X, mask=selector.get_support())

def select_features():
    label='Vote'
    #   1: Load data
    elections_df = read_csv("ElectionsData.csv", header=0)
    ## labeled samples only
    elections_df = elections_df[elections_df[label].notna()]
    #   2: Set the correct type of each attribute
    set_types(elections_df)
    #   3: Data preparation
    #   3.1: Imputation
    elections_df = closet_fit(elections_df)
    #   3.2: Data Cleansing
    elections_df = noise_outlier_variance(elections_df, k=2.5)
    elections_df = categorical_to_numerical(elections_df, label)
    #   3.3: Normalization (scaling)
    X, y = (elections_df.drop(label, axis=1), elections_df[label])
    X = scale_data(X)
    #   3.4: Feature Selection
    corr = X.corr()
    features = list(X.columns)
    features_to_remove = []
    for feat1 in corr.columns:
        for feat2 in corr.columns:
            if feat1 == feat2 or features.count(feat1) == 0 or features.count(feat2) == 0:
                continue
            if np.abs(corr[feat1][feat2]) > 0.95:
                features.remove(feat1)
                features_to_remove.append(feat1)
    X.drop(features_to_remove, axis=1, inplace=True)
    ## After this step there are 50 features in hand
    ## Filter method feature selection
    ## Choose k=35 best by SelectKBest
    filtered_features = feature_select_filter(X, y, k=35)
    features_to_remove = list(set(X.columns).difference(filtered_features))
    X.drop(features_to_remove, axis=1, inplace=True)
    ## Wrapper method using Decision Trees
    filtered_features = feature_select_SVM_wrapper(X, y)
    features_to_remove = list(set(X.columns).difference(filtered_features))
    X.drop(features_to_remove, axis=1, inplace=True)

    selected_features = set([feature.split('-')[0] for feature in X.columns])
    #print('Selected my mine: ', selected_features)
    out = read_csv('SelectedFeatures.csv', header=0, index_col=0)
    for feature in selected_features:
        out['Used?': feature] = int(True)
    out.to_csv('SelectedFeatures.csv')

    X_train, y_train, X_valid, y_valid, X_test, y_test = split_dataset(X, y)
    exit(0)


#################################
#       Script starts here      #
#################################
def preprocess():
    label='Vote'
    best_features = [ "Yearly_IncomeK", "Number_of_differnt_parties_voted_for",
                      "Political_interest_Total_Score", "Avg_Satisfaction_with_previous_vote",
                      "Avg_monthly_income_all_years", "Most_Important_Issue",
                      "Overall_happiness_score", "Avg_size_per_room", "Weighted_education_rank"]
    #   1: Load data
    elections_df = read_csv("ElectionsData.csv", header=0)
    ## labeled samples only
    elections_df = elections_df[elections_df[label].notna()]
    #   2: Set the correct type of each attribute
    set_types(elections_df)

    #best_features = list(elections_df.keys())
    #best_features.remove(label)
    X, y = elections_df[best_features],elections_df[label]

    X_train, y_train, X_valid, y_valid, X_test, y_test = split_dataset(X,y)
    #   3: Data preparation
    #   3.1 Imputing
    X_train = closet_fit(X_train)
    X_test = closet_fit(X_test)
    X_valid = closet_fit(X_valid)

    print('cat to num')
    X_train = categorical_to_numerical(X_train, y_train)
    X_test = categorical_to_numerical(X_test, y_test)
    X_valid = categorical_to_numerical(X_valid, y_valid)

    #   3.2: Data Cleansing
    print(remove_noise_outliers(X_train, y_train))
    print(remove_noise_outliers(X_test, y_test))
    print(remove_noise_outliers(X_valid, y_valid))

    X_train = remove_noise_outliers(X_train, y_train)
    X_test = remove_noise_outliers(X_test, y_test)
    X_valid = remove_noise_outliers(X_valid, y_valid)

    #   3.3: Normalization (scaling)
    X_train = scale_data(X_train)
    X_test = scale_data(X_test)
    X_valid = scale_data(X_valid)

    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)

    """X_train = categorical_to_numerical(X_train, y_train)
    X_test = categorical_to_numerical(X_test, y_test)
    X_valid = categorical_to_numerical(X_valid, y_valid)"""
    print(X_train,y_train)
    X_train.to_csv('x_train.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    X_valid.to_csv('x_valid.csv', index=False)
    y_valid.to_csv('y_valid.csv', index=False)
    X_test.to_csv('x_test.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

if __name__ == '__main__':
    preprocess()