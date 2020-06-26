import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, flash
import os

#

flask_app = Flask(__name__, static_url_path='/static')

# ML model path
model_path = "Models/model.pkl"

data = pd.read_csv("Models/bank-full.csv", sep=";",header='infer')

model = pickle.load(open(model_path, 'rb'))


@flask_app.route('/')
def index_page():
    return render_template('index.html')



@flask_app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [int_features]
    prediction = model.predict(final_features)

    output = prediction[0]
    Answer = ''
    if output == 1:
        Answer = "Yes"

    else:
        Answer = "No"
    #conf_score = prediction.predict_proba([final_features])* 100


    return render_template('index.html', prediction_text='Is the customer Going to subscribe to a term deposit? {}, final features {}'.format(Answer, final_features))


if __name__ == "__main__":
    flask_app.run(host='0.0.0.0', port=5000, debug=True)
