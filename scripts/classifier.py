# -*- coding: utf-8 -*-
import datetime

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model.perceptron import Perceptron
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.linear_model.logistic import LogisticRegression as MaxEnt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from scipy import io

from samples import NB_CLASSIFIER, CLEARED_DATA_NB_CLASSIFIER, REAL_TEST_FEATURES_FILE_NAME


POSITIVE_CLASS = "pos"
NEGATIVE_CLASS = "neg"
# NEUTRAL_CLASS = "neut"


# classifiers start


# done
def fit_svc(feature_matrix, answers):
    svc = LinearSVC(loss='l2', dual=False, tol=1e-3)
    return svc.fit(feature_matrix, answers)


# done
def fit_sgd(feature_matrix, answers):
    svc = SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")
    return svc.fit(feature_matrix, answers)


# done
def fit_nb(feature_matrix, answers):
    gnb = MultinomialNB()
    return gnb.fit(feature_matrix, answers)


# done
def fit_per(feature_matrix, answers):
    per = Perceptron()
    return per.fit(feature_matrix, answers)


# done
def fit_pac(feature_matrix, answers):
    pac = PassiveAggressiveClassifier()
    return pac.fit(feature_matrix, answers)


# done
def fit_mxe(feature_matrix, answers):
    mxe = MaxEnt()
    return mxe.fit(feature_matrix, answers)


# sucks
def fit_rfc(feature_matrix, answers):
    rfc = RandomForestClassifier()
    return rfc.fit(feature_matrix, answers)


# sucks
def fit_rid(feature_matrix, answers):
    rid = RidgeClassifier(tol=1e-2, solver="lsqr")
    return rid.fit(feature_matrix, answers)


# sucks
def fit_knn(feature_matrix, answers):
    knn = KNeighborsClassifier()
    return knn.fit(feature_matrix, answers)


# classifiers end


def read_feature_matrix_from_file(file_path):
    return io.mmread(file_path)


def generate_training_answers(n):
    return np.concatenate((np.array([POSITIVE_CLASS] * n), np.array([NEGATIVE_CLASS] * n)))
    # np.array([NEUTRAL_CLASS] * n)


def generate_test_answers(m):
    return np.concatenate((np.array([POSITIVE_CLASS] * m), np.array([NEGATIVE_CLASS] * m)))
    # np.array([NEUTRAL_CLASS] * m) POS = 235, NEG = 699, NEUT = 1066


def generate_real_test_answers():
    return np.concatenate((np.array([POSITIVE_CLASS] * 235), np.array([NEGATIVE_CLASS] * 699)))


def evaluate(classifier, test_features, answers):
    predicted = classifier.predict(test_features)
    target_names = [NEGATIVE_CLASS, POSITIVE_CLASS]
    # NEUTRAL_CLASS
    print(classification_report(answers, predicted, target_names=target_names))
    print("accuracy : %s" % accuracy_score(answers, predicted))


if __name__ == "__main__":
    time = datetime.datetime.now()
    # classifier = fit_svc(read_feature_matrix_from_file(TRAIN_FEATURES_FILE_NAME), generate_training_answers())

    # classifier = fit_nb(read_feature_matrix_from_file(TRAIN_FEATURES_FILE_NAME), generate_training_answers(4000000))

    # classifier = fit_nb(read_feature_matrix_from_file(TRAIN_FEATURES_FILE_NAME), generate_training_answers(2500000))

    # joblib.dump(classifier, CLEARED_DATA_NB_CLASSIFIER)

    classifier = joblib.load(CLEARED_DATA_NB_CLASSIFIER)

    print("fit model: seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds())
    time = datetime.datetime.now()
    # test_features = read_feature_matrix_from_file(TEST_FEATURES_FILE_NAME)
    test_features = read_feature_matrix_from_file(REAL_TEST_FEATURES_FILE_NAME)
    # answers = generate_test_answers(100000)
    answers = generate_real_test_answers()
    evaluate(classifier, test_features, answers)
    print("evaluate answers: seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds())
