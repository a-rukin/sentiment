# -*- coding: utf-8 -*-
from pip._vendor import requests
from pymystem3 import Mystem
# import pymorphy2
import re
import datetime

# FEATURE_EXTRACTORS

from samples import POS_TWEETS_FILE_ONLY_NAME, NEG_TWEETS_FILE_ONLY_NAME, DIR_TWEETS_FILES

smiles = [':)', ':-)', ': )', ':d', '=)', '))', ':(', ':-(', ': (', '=(', '((', ')', '(']
# morph = pymorphy2.MorphAnalyzer()
mystem = Mystem()
DICTIONARY = {}


# удаляем лишние пробелы
def tweet_strip(line):
    return ' '.join(line.split())


# удаляем смайлы, т.к. по ним проходила предварительная выборка
def delete_smile(line):
    for s in smiles:
        line = line.replace(s, '')
    return line


# убираем имя пользователя и ссылки на другие имена
def clear_of_name(line):
    # имя пользователя идет первым, поэтому просто удаляем всё до пробела
    s, t, line = line.partition(" ")
    #удаляем все ссылки на другие имена
    return re.sub(r"@[^\s]+", " ", line)


# убираем ссылки
def clear_of_link(line):
    #удаляем все ссылки
    return re.sub(r"http[^\s]+", " ", line)


#удаляем цифры
def delete_digits(line):
    trans_dict = {ord("{}".format(x)): "" for x in range(10)}
    return line.translate(trans_dict)


#удаляем повторения букв "аааа"
def delete_repeat(line):
    while bool(re.compile(r"([a-zA-Zа-яА-Я])\1\1").search(line)):
        line = re.sub(r"([a-zA-Zа-яА-Я])\1\1", r"\1\1", line)
    return line


# удаляем все кроме пробелов и букв (удаленное заменяем одинарными пробелами)
def delete_non_letter(line):
    return re.sub(r"[^\s\w]+|\d+", " ", line)


def normalized_word(word):
    if DICTIONARY.get(word):
        return DICTIONARY.get(word)
    else:
        # normal_form = morph.parse(word)[0].normal_form
        normal_form = mystem.lemmatize(word)[0]
        DICTIONARY[word] = normal_form
        return normal_form


def check_grammar(word):
    params = {'text': word, 'lang': 'ru'}
    # r = requests.get('http://speller.yandex.net/services/spellservice.json/checkText', params=params)
    r = requests.get('http://erratum-test.yandex.ru:19056/spellservice.json/checkText', params=params)

    if r.status_code == 200:
        if len(r.json()) > 0:
            out = r.json()[0]
            variants = [v for v in out['s']]
            if len(variants) is not 0:
                return variants[0].split(" ")[0]
            else:
                return word
        else:
            return word


def normalized_tweet(tweet):
    return " ".join(map(lambda w: normalized_word(w), tweet.split()))


def clean_tweet(tweet):
    return normalized_tweet(
        delete_repeat(tweet_strip(delete_non_letter(delete_smile(clear_of_link(clear_of_name(tweet_strip(tweet.lower()))))))))


def clean_tweets_without_dubl(path, filename):
    txt_file = open(path + filename, encoding="utf8")
    txt_clean = open(''.join([path, 'clean_', filename]), 'w+', encoding="utf8")
    txt_featured = open(''.join([path, 'featured_', filename]), 'w+', encoding="utf8")

    lines = set()
    ordered_lines = []
    old_ordered_lines = []
    line_number = 0
    clear_time = datetime.datetime.now()

    for line in txt_file:
        line_number += 1
        old_line = line
        line = clean_tweet(line)
        if len(line):
            if line not in lines:
                ordered_lines.append(line)
                old_ordered_lines.append(old_line)
            lines.add(line)
        if line_number % 10000 == 0:
            print("%s was cleared" % line_number)
            print("seconds passed: %s" % (datetime.datetime.now() - clear_time).total_seconds())
            clear_time = datetime.datetime.now()
    print("%s total was cleared" % line_number)

    for i, line in enumerate(ordered_lines):
        txt_clean.write(line + '\n')
        if old_ordered_lines[i][-1] == '\n':
            new_line = ''
        else:
            new_line = '\n'
        txt_featured.write(old_ordered_lines[i] + new_line)
    print("%s total was written" % len(ordered_lines))


if __name__ == '__main__':
    time = datetime.datetime.now()
    clean_tweets_without_dubl(DIR_TWEETS_FILES, POS_TWEETS_FILE_ONLY_NAME)
    print("positive tweets cleaning: seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds())
    time = datetime.datetime.now()
    clean_tweets_without_dubl(DIR_TWEETS_FILES, NEG_TWEETS_FILE_ONLY_NAME)
    print("negative tweets cleaning: seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds())

    a = clean_tweet("ололоев превед!! мой свет!")
    print(a)
    a = clean_tweet(
        "savva601 сломленных	ааааа @KSyomin  ага, а перед разобранным мостом ещё лежит, для ускорения, огромная куча дерьма в виде... а под мостом кол в виде БРИКС и ШОС! :-))")
    print(a)
    a = clean_tweet("barabanlyna	@I_need_a_beach о да\nЭто жопа((")
    print(a)
