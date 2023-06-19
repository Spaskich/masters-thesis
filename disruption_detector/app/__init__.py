import os
from flask import Flask, request
from tetracomnlp.ml import BinaryLogisticClassifier

from app.scripts.helper import get_sentence_texts, get_classifier


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # initialize the classifier
    binary_logistic_classifier = get_classifier()

    # print on server running
    print("Disruption Detector Version 0.0.1")

    # Test if server is running
    @app.route('/test')
    def test():
        return 'Server is running.'

    # Classification workflow
    @app.route('/classification', methods=['GET', 'POST'])
    def classification():
        json_data = request.json
        documents = json_data['documents']
        output_json = {
            'documents': []
        }

        if len(documents) == 0:
            return {
                'exception': 'No documents!'
            }
        else:
            for document_index, document in enumerate(documents):
                print(f'Processing document #{document_index + 1} out of {len(documents)}...')

                sentences = get_sentence_texts(document['sentences'])
                prediction = 0
                shifts = []

                try:
                    document_representation = binary_logistic_classifier.get_document_representation(sentences)
                    # prediction = binary_logistic_classifier.get_prediction(document_representation)
                    prediction, shifts = binary_logistic_classifier.get_statistics(document_representation)
                except ValueError as e:
                    print(f'Error processing document #{document_index}. Error: {e}')
                    print(f'Sentences: {sentences}')

                document['prediction'] = prediction
                document['shifts'] = shifts.tolist()
                output_json['documents'].append(document)

            return output_json

        # elif len(documents) == 1:
        #     document = documents[0]
        #     sentences = document['sentences']
        #
        #     document_representation = binary_logistic_classifier.get_document_representation(sentences)
        #     prediction = binary_logistic_classifier.get_prediction(document_representation)
        #
        #     return {'prediction': prediction}
        # else:
        #     document_representations = np.zeros((len(documents), binary_logistic_classifier.dimensionality),
        #                                         dtype=binary_logistic_classifier._modeler.data_type)
        #
        #     for document_index, document in enumerate(documents):
        #         print(f'Processing document #{document_index}...')
        #
        #         sentences = document['sentences']
        #
        #         try:
        #             document_representation = binary_logistic_classifier.get_document_representation(sentences)
        #             document_representations[document_index] = document_representation
        #         except ValueError:
        #             print(f'Error processing document #{document_index}.')
        #             print(f'Sentences: {sentences}')
        #
        #     predictions = binary_logistic_classifier.get_predictions(document_representations)
        #     print(predictions.shape)
        #
        #     return {'predictions': predictions.tolist()}

    # Process Solr JSONs for classification
    @app.route('/processSolrJson')
    def process_solr_json():
        json_data = request.json

        documents = {
            "model_name": "RoBERTa",
            "strategy": 3,
            'documents': []
        }

        sentences = []
        for doc in json_data['docs']:
            document_uuid = doc['factor_text_item_uid']
            sentence_uuid = doc['id']

            document_title = doc['en_a_title'][0]
            sentence_text = doc['en_a_text'][0]

            sentence_index = doc['factor_int_seq']

            if sentence_index < 4:
                if sentence_index == 0:
                    sentences = [{
                        'uuid': f'{document_uuid}_title',
                        'text': document_title
                    }]

                sentences.append({
                    'uuid': sentence_uuid,
                    'text': sentence_text
                })

                if sentence_index == 3 and len(sentences) > 0:
                    documents['documents'].append(
                        {
                            'uuid': document_uuid,
                            'sentences': sentences
                        })

        return documents

    return app
