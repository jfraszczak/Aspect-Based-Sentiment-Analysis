import ABSA
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

global model1
model1 = ABSA.load_model('models/BiLSTM_Aspect.h5')

global model2
model2 = ABSA.load_model('models/BiLSTM_Sentiment.h5')


@app.route("/", methods=['GET', 'POST'])
def main():
    text = ''
    aspects = []
    input = False

    if request.method == 'POST':
        text = request.form.get('form')

        global model1
        global model2
        aspects = ABSA.extract_sentiment(text, model1, model2)
        input = True

    return render_template('index.html', text=text, aspects=aspects, input=input)


if __name__ == "__main__":
    app.run(debug=False, threaded=False)
