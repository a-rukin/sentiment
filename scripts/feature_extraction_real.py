# -*- coding: utf-8 -*-
import datetime
import numpy as np
import scipy.sparse as sp
from scipy import sparse, io
from sklearn.feature_extraction.text import CountVectorizer
from feature_extraction import extract_and_dump_features as extract_and_dump_real_features, clean_tweets, compose_data

from samples import REAL_POS_TWEETS_FILE_NAME, REAL_NEG_TWEETS_FILE_NAME, \
    REAL_UNIGRAMS_TEST_FEATURES_FILE_NAME, REAL_BIGRAMS_TEST_FEATURES_FILE_NAME, REAL_TEST_OTHER_FEATURES_FILE_NAME, \
    REAL_TEST_FEATURES_FILE_NAME, REAL_TEST_TWEETS_FILE_NAME, STOP_WORDS, \
    CLEAN_TRAIN_TWEETS_FILE_NAME


def make_real_test_set(pos_tweets, neg_tweets, output_test_file):
    added_tweets = 0
    for tweet in pos_tweets:
        added_tweets += 1
        output_test_file.write(tweet)
    print("positive tweets added - %d", added_tweets)
    added_tweets = 0
    for tweet in neg_tweets:
        added_tweets += 1
        output_test_file.write(tweet)
    print("negative tweets added - %d", added_tweets)


def extract_uni_features_from_real_tweets(train_tweets, test_tweets, output_test_file_path):
    stop_words = [line.rstrip() for line in open(STOP_WORDS)]
    print(stop_words)
    vector_sk = CountVectorizer(min_df=1, stop_words=stop_words)
    vector_sk.fit_transform(train_tweets)
    test_matrix = vector_sk.transform(test_tweets)
    print(len(vector_sk.get_feature_names()))
    io.mmwrite(output_test_file_path, test_matrix)
    print("unigrams test matrix dumped")


def extract_bigram_features_from_real_tweets(train_tweets, test_tweets, output_test_file_path):
    vector_sk = CountVectorizer(ngram_range=(2, 2), min_df=1)
    vector_sk.fit_transform(train_tweets)
    test_matrix = vector_sk.transform(test_tweets)
    print(len(vector_sk.get_feature_names()))
    io.mmwrite(output_test_file_path, test_matrix)
    print("bigrams test matrix dumped")


def extract_and_dump_real_ngram_features(input_train_file_path, input_test_file_path):
    clean_training_tweets = compose_data(open(input_train_file_path))
    clean_test_tweets = clean_tweets(open(input_test_file_path))
    extract_uni_features_from_real_tweets(clean_training_tweets, clean_test_tweets,
                                          REAL_UNIGRAMS_TEST_FEATURES_FILE_NAME)
    extract_bigram_features_from_real_tweets(clean_training_tweets, clean_test_tweets,
                                             REAL_BIGRAMS_TEST_FEATURES_FILE_NAME)


def make_all_real_features_matrix(unigrams_file_path, bigrams_file_path, other_features_file_path,
                                  all_features_file_path):
    sparse_features_matrix = sparse.csr_matrix(np.loadtxt(other_features_file_path))
    unigrams_matrix = io.mmread(unigrams_file_path)
    bigrams_matrix = io.mmread(bigrams_file_path)
    all_features = sp.hstack([unigrams_matrix, bigrams_matrix, sparse_features_matrix])
    io.mmwrite(all_features_file_path, all_features)


if __name__ == '__main__':
    make_real_test_set(open(REAL_POS_TWEETS_FILE_NAME, encoding="utf8"),
                       open(REAL_NEG_TWEETS_FILE_NAME, encoding="utf8"),
                       open(REAL_TEST_TWEETS_FILE_NAME, 'w'))

    time = datetime.datetime.now()

    extract_and_dump_real_features(REAL_TEST_TWEETS_FILE_NAME, REAL_TEST_OTHER_FEATURES_FILE_NAME)

    extract_and_dump_real_ngram_features(CLEAN_TRAIN_TWEETS_FILE_NAME, REAL_TEST_TWEETS_FILE_NAME)

    make_all_real_features_matrix(REAL_UNIGRAMS_TEST_FEATURES_FILE_NAME, REAL_BIGRAMS_TEST_FEATURES_FILE_NAME,
                                  REAL_TEST_OTHER_FEATURES_FILE_NAME, REAL_TEST_FEATURES_FILE_NAME)

    print("extract features: seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds())
