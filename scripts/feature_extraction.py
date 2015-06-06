# -*- coding: utf-8 -*-
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp
from scipy import sparse, io
import datetime

from samples import TEST_OTHER_FEATURES_FILE_NAME, TRAIN_OTHER_FEATURES_FILE_NAME, TEST_FEATURES_FILE_NAME,\
    TRAIN_FEATURES_FILE_NAME, \
    UNIGRAMS_TEST_FEATURES_FILE_NAME, UNIGRAMS_TRAIN_FEATURES_FILE_NAME, BIGRAMS_TEST_FEATURES_FILE_NAME, \
    BIGRAMS_TRAIN_FEATURES_FILE_NAME, \
    STOP_WORDS, \
    FEATURED_POS_TWEETS_FILE_NAME, FEATURED_NEG_TWEETS_FILE_NAME, \
    CLEAN_POS_TWEETS_FILE_NAME, CLEAN_NEG_TWEETS_FILE_NAME, \
    FEATURED_TEST_TWEETS_FILE_NAME, FEATURED_TRAIN_TWEETS_FILE_NAME, \
    CLEAN_TEST_TWEETS_FILE_NAME, CLEAN_TRAIN_TWEETS_FILE_NAME

from tweet_prepare import clean_tweet

# FEATURE_EXTRACTORS
NO_WORD_REGEXP = re.compile(r"\bне\b")
URL_REGEXP = re.compile(r"http(s)?://")
BIG_WORDS_REGEXP = re.compile(r"\b[^a-zA-Zа-я\W\s\d]+\b")
REPEAT_LETTERS_REGEXP = re.compile(
    r"([a-zA-Zа-яА-Я])\1\1|([a-zA-Zа-яА-Я][^\s\d])\2\2|([a-zA-Zа-яА-Я][^\s\d]{2})\3|([a-zA-Zа-яА-Я][^\s\d]{3})\4")
REPEAT_EXCLAMATION_MARK = re.compile(r"!(!+|1{2})")
MENTION_REGEXP = re.compile(r"\b@")


def has_exclamation_mark(tweet):
    return int('!' in tweet)


def has_repeat_exclamation_mark(tweet):
    return int(bool(REPEAT_EXCLAMATION_MARK.search(tweet)))


def has_question_mark(tweet):
    return int('?' in tweet)


def has_word_not(tweet):
    return int(bool(NO_WORD_REGEXP.search(tweet)))


def has_url(tweet):
    return int(bool(URL_REGEXP.search(tweet)))


def has_big_word(tweet):
    return int(bool(BIG_WORDS_REGEXP.search(tweet)))


def has_repeat_letters(tweet):
    return int(bool(REPEAT_LETTERS_REGEXP.search(tweet)))


def has_mention(tweet):
    return int(bool(MENTION_REGEXP.search(tweet)))


FEATURE_EXTRACTORS = [has_word_not, has_exclamation_mark, has_repeat_exclamation_mark,
                      has_question_mark, has_url, has_big_word, has_repeat_letters, len, has_mention]
# FEATURE EXTRACTION


def extract_features_from_tweet(tweet):
    return np.array([extractor(tweet) for extractor in FEATURE_EXTRACTORS])


def extract_features_from_tweets(tweets, output_file):
    tweet_number = 0
    for tweet in tweets:
        tweet_number += 1
        output_file.write(" ".join(map(str, extract_features_from_tweet(tweet))))
        output_file.write("\n")
        if tweet_number % 10000 == 0:
            print(tweet_number)


def extract_and_dump_features(input_file_path, output_file_path):
    extract_features_from_tweets(open(input_file_path, encoding="utf8"), open(output_file_path, "w"))


def delete_nickname(tweet):
    try:
        return tweet.split("\t")[1]
    except IndexError:
        return tweet


def clean_tweets(tweets):
    cleaned_tweets = []
    for i, tweet in enumerate(tweets.readlines()):
        cleaned_tweets.append(clean_tweet(tweet))
        if i % 10000 == 0:
            print("cleaned {} tweets".format(i))
    print("Total: cleaned {} tweets".format(len(cleaned_tweets)))
    return cleaned_tweets


def compose_data(tweets):
    result_tweets = []
    for tweet in tweets.readlines():
        result_tweets.append(tweet)
    return result_tweets


def extract_uni_features_from_tweets(tweets, test_tweets, output_training_file_path, output_test_file_path):
    stop_words = [line.rstrip() for line in open(STOP_WORDS)]
    print(stop_words)
    vector_sk = CountVectorizer(min_df=1, stop_words=stop_words)
    features_matrix = vector_sk.fit_transform(tweets)
    test_matrix = vector_sk.transform(test_tweets)
    print(len(vector_sk.get_feature_names()))
    io.mmwrite(output_training_file_path, features_matrix)
    print("unigrams training matrix dumped")
    io.mmwrite(output_test_file_path, test_matrix)
    print("unigrams test matrix dumped")


def extract_bigram_features_from_tweets(tweets, test_tweets, output_training_file_path, output_test_file_path):
    vector_sk = CountVectorizer(ngram_range=(2, 2), min_df=1)
    features_matrix = vector_sk.fit_transform(tweets)
    test_matrix = vector_sk.transform(test_tweets)
    print(len(vector_sk.get_feature_names()))
    io.mmwrite(output_training_file_path, features_matrix)
    print("bigrams training matrix dumped")
    io.mmwrite(output_test_file_path, test_matrix)
    print("bigrams test matrix dumped")


def extract_four_gram_features_from_tweets(tweets, test_tweets, output_training_file_path, output_test_file_path):
    vector_sk = CountVectorizer(ngram_range=(4, 4), analyzer="char", min_df=1)
    features_matrix = vector_sk.fit_transform(tweets)
    test_matrix = vector_sk.transform(test_tweets)
    print(len(vector_sk.get_feature_names()))
    io.mmwrite(output_training_file_path, features_matrix)
    print("four grams training matrix dumped")
    io.mmwrite(output_test_file_path, test_matrix)
    print("four grams test matrix dumped")


def extract_and_dump_ngram_features(input_file_path, input_test_file_path):
    clean_training_tweets = compose_data(open(input_file_path))
    clean_test_tweets = compose_data(open(input_test_file_path))
    extract_uni_features_from_tweets(clean_training_tweets, clean_test_tweets,
                                     UNIGRAMS_TRAIN_FEATURES_FILE_NAME, UNIGRAMS_TEST_FEATURES_FILE_NAME)
    # extract_four_gram_features_from_tweets(clean_training_tweets, clean_test_tweets,
    #                                FOUR_GRAMMS_TRAIN_FEATURES_FILE_NAME, FOUR_GRAMMS_TEST_FEATURES_FILE_NAME)
    extract_bigram_features_from_tweets(clean_training_tweets, clean_test_tweets,
                                        BIGRAMS_TRAIN_FEATURES_FILE_NAME, BIGRAMS_TEST_FEATURES_FILE_NAME)


def make_test_and_training_set(pos_tweets, neg_tweets, output_training_file, output_test_file,
                               training_set_size, test_set_size):
    added_tweets = 0
    for tweet in pos_tweets:
        added_tweets += 1
        if training_set_size >= added_tweets:
            output_training_file.write(tweet)
        else:
            if training_set_size + test_set_size >= added_tweets:
                output_test_file.write(tweet)
            else:
                print(added_tweets)
                break
    print("length %d", training_set_size)
    added_tweets = 0
    for tweet in neg_tweets:
        added_tweets += 1
        if training_set_size >= added_tweets:
            output_training_file.write(tweet)
        else:
            if training_set_size + test_set_size >= added_tweets:
                output_test_file.write(tweet)
            else:
                print(added_tweets)
                break


def make_all_features_matrix(unigrams_file_path, bigrams_file_path,
                             other_features_file_path, all_features_file_path):
    sparse_features_matrix = sparse.csr_matrix(np.loadtxt(other_features_file_path))
    unigrams_matrix = io.mmread(unigrams_file_path)
    bigrams_matrix = io.mmread(bigrams_file_path)
    all_features = sp.hstack([unigrams_matrix, bigrams_matrix, sparse_features_matrix])
    io.mmwrite(all_features_file_path, all_features)


if __name__ == '__main__':
    # Создаем обучающую и тестовую выборку
    TRAINING_SET_SIZE = 2500000
    TEST_SET_SIZE = 100000
    make_test_and_training_set(open(CLEAN_POS_TWEETS_FILE_NAME, encoding="utf8"),
                               open(CLEAN_NEG_TWEETS_FILE_NAME, encoding="utf8"),
                               open(CLEAN_TRAIN_TWEETS_FILE_NAME, 'w+'),
                               open(CLEAN_TEST_TWEETS_FILE_NAME, 'w+'),
                               TRAINING_SET_SIZE, TEST_SET_SIZE)
    make_test_and_training_set(open(FEATURED_POS_TWEETS_FILE_NAME, encoding="utf8"),
                               open(FEATURED_NEG_TWEETS_FILE_NAME, encoding="utf8"),
                               open(FEATURED_TRAIN_TWEETS_FILE_NAME, 'w+'),
                               open(FEATURED_TEST_TWEETS_FILE_NAME, 'w+'),
                               TRAINING_SET_SIZE, TEST_SET_SIZE)
    time = datetime.datetime.now()
    # Выделяем наши факторы
    extract_and_dump_features(FEATURED_TRAIN_TWEETS_FILE_NAME, TRAIN_OTHER_FEATURES_FILE_NAME)
    extract_and_dump_features(FEATURED_TEST_TWEETS_FILE_NAME, TEST_OTHER_FEATURES_FILE_NAME)
    # Выделяем N-граммы
    extract_and_dump_ngram_features(CLEAN_TRAIN_TWEETS_FILE_NAME, CLEAN_TEST_TWEETS_FILE_NAME)
    # Склеиваем матрицу факторов и матрицы n-грамм.
    make_all_features_matrix(UNIGRAMS_TEST_FEATURES_FILE_NAME,
                             BIGRAMS_TEST_FEATURES_FILE_NAME,
                             TEST_OTHER_FEATURES_FILE_NAME,
                             TEST_FEATURES_FILE_NAME)
    print("test features matrix made")
    make_all_features_matrix(UNIGRAMS_TRAIN_FEATURES_FILE_NAME,
                             BIGRAMS_TRAIN_FEATURES_FILE_NAME,
                             TRAIN_OTHER_FEATURES_FILE_NAME,
                             TRAIN_FEATURES_FILE_NAME)
    print("training features matrix made")
    print("extract features: seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds())
