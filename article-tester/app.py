import json

from flask import Flask, render_template, request, redirect

from scripts.helper import get_classifier

app = Flask(__name__)

# Initialize the Classifier
binary_logistic_classifier = get_classifier()


def analyze_sentences(sentences):
    modeler = binary_logistic_classifier.modeler

    sentence_analysis = []

    for sentence in sentences:
        tokens, is_valid_token, output = modeler.run_on_sentence(sentence)
        sentence_analysis.append({
            'text': sentence,
            'tokens': tokens,
            'is_valid_token': is_valid_token
        })

    document_representation = binary_logistic_classifier.get_document_representation(sentences)
    prediction, shifts = binary_logistic_classifier.get_statistics(document_representation)

    return {
        'prediction': prediction,
        'shifts': shifts,
        'sentence_analysis': sentence_analysis
    }


@app.route('/', methods=['GET', 'POST'])
def single():
    number_of_sentences = binary_logistic_classifier.sentences_per_document
    return render_template('single_article_form.html', number_of_sentences=number_of_sentences)


@app.route('/multiple', methods=['GET', 'POST'])
def multiple():
    return render_template('multiple_articles_form.html')


@app.route('/success-single', methods=['GET', 'POST'])
def success_single():
    if request.method == 'POST':
        sentences = []

        for i in range(binary_logistic_classifier.sentences_per_document):
            sentence = request.form['sentence{}'.format(i)]
            if sentence:
                sentences.append(sentence)

        document_analysis = analyze_sentences(sentences)
        document_analyses = [document_analysis]

        return render_template('analysis.html', document_analyses=document_analyses,
                               strategy=binary_logistic_classifier.strategy,
                               model_name=binary_logistic_classifier.modeler.name)
    else:
        # Redirect to the form page if accessed directly without form submission
        return redirect('/')


@app.route('/success-multiple', methods=['GET', 'POST'])
def success_multiple():
    if request.method == 'POST':
        document_analyses = []

        json_data = request.form['jsonData']  # Get the JSON data from the form
        data = json.loads(json_data)  # Parse the JSON string

        documents = data['documents']  # Get the 'documents' array from the parsed JSON

        for document in documents:
            sentences = [sentence['text'] for sentence in
                         document['sentences']]  # Extract only the 'text' field for each sentence

            # Call the 'analyze_sentences' method for each document
            document_analyses.append(analyze_sentences(sentences))

        return render_template('analysis.html', document_analyses=document_analyses,
                               strategy=binary_logistic_classifier.strategy,
                               model_name=binary_logistic_classifier.modeler.name)
    else:
        # Redirect to the form page if accessed directly without form submission
        return redirect('/multiple')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
