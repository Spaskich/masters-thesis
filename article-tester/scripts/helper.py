import os

import requests
from requests.auth import HTTPBasicAuth
from tetracomnlp.ml import BinaryLogisticClassifier


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
