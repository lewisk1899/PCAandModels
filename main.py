import numpy as np
import pandas as pd
import sklearn as skl
from sklearn import svm
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn import decomposition
import time
from tqdm import tqdm
from joblib import dump, load


def data_processing(filename):
    # The data we are processing is a covid classification dataset
    # where 20 features are analyzed to determine the likelihood of
    # dying from covid-19
    covid_data = pd.read_csv(filename)
    # converting from date died to whether or not dead
    covid_data['DATE_DIED'] = (covid_data['DATE_DIED'] != '9999-99-99').astype(int)
    # splitting the data into a training and testing set, randomly shuffled for cross validation
    train, test = train_test_split(covid_data, shuffle=True, train_size=.8)
    # need to convert train/test data frame into (x_train, y_train) and (x_test, y_test)
    # removing the target for the training and testing
    train_target = train.pop("DATE_DIED").to_numpy()[:40000]
    test_target = test.pop("DATE_DIED").to_numpy()[:10000]
    # converting df to list
    train_np = train.to_numpy()[:40000]
    test_np = test.to_numpy()[:10000]

    return (train_np, train_target), (test_np, test_target)


def PCA(X):
    pca = decomposition.PCA(n_components=10)
    transformed_data = pca.fit_transform(X)
    print(transformed_data)


# Second Model: Support Vector Machine
# Assumption Made: Assume the data is linearly separable.
def SVM(train):
    clf = svm.SVC()
    start = time.time()  # time how long it take for the model to train
    clf.fit(train[0], train[1])
    stop = time.time()  # stop timing
    print(f"Training time: {stop - start}s")
    dump(clf, 'SVMweights.joblib')


def get_model():
    clf = load('SVMweights.joblib')
    return clf


# clf = get_model()


def score_SVM(clf, X, y):
    # cross validating to see accuracy
    scores = skl.model_selection.cross_val_score(clf, X, y, cv=5)
    print('Support Vector Machine Cross-Validation')
    print(scores)


# Three Models:

# First Model: Multi-Class Naive Bayes
# Naive Bayes Model: Posterior Probability = Likelihood*Prior/Evidence
# Maybe cannot use naive bayes as it assumes that features are independent, possibly find another dataset for this.

def naive_bayes(X, y):
    model = CategoricalNB()
    model.fit(X, y)
    return model


# NB_model = naive_bayes(train_np, train_target)


def score_naive_bayes(model, X, y):
    scores = skl.model_selection.cross_val_score(model, X, y, cv=5)
    print('Naive Bayes Cross-Validation')
    print(scores)


# score_naive_bayes(NB_model, train_np, train_target)

# Second Model: Support Vector Machine
# Third Model: Decision Trees/Random Forest


def train_models():
    (train_np, train_target), (test_np, test_target) = data_processing('Covid Data.csv')
    SVM((train_np, train_target))  # training the model
    naive_bayes(train_np, train_target)


train_models()
