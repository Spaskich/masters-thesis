import json
import os

import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from tetracomnlp.ml import BinaryLogisticClassifier


def array2string(values: np.ndarray):
    return '[' + ','.join([str(x) for x in values]) + ']'


def model_info2json(model_name: str, strategy: int, sentences: int, classifier,
                    good_dimensions: np.ndarray, separator: str = '\n'):
    indentation = '' if separator == '' else '\t'
    result = '{' + separator
    result += f'{indentation}"model_name": "{model_name}",{separator}'
    result += f'{indentation}"strategy": {strategy},{separator}'
    result += f'{indentation}"sentences": {json.dumps(sentences)},{separator}'
    result += indentation + '"indices": ' + array2string(good_dimensions) + ',' + separator
    result += indentation + '"coefficients": ' + array2string(classifier.coef_[0, :]) + ',' + separator
    result += indentation + '"intercept": ' + str(classifier.intercept_[0]) + separator + '}'

    return result


def get_sentence_texts(sentences: json):
    sentence_texts = []

    for sentence in sentences:
        sentence_texts.append(sentence['text'])
    print(sentences)
    print(sentence_texts)

    return sentence_texts


def get_classifier():
    bitbucket_url = os.getenv('REPOSITORY_URL')
    classifier_path = os.getenv("CLASSIFIER_PATH")
    classifier_url = f'{bitbucket_url}/raw/{classifier_path}'

    username = os.getenv('BITBUCKET_USER')
    password = os.getenv('BITBUCKET_PASSWORD')

    response = requests.get(classifier_url, auth=HTTPBasicAuth(username, password))
    json_response = response.json()

    print(json_response)

    return BinaryLogisticClassifier(json_response)

