import os
import pickle
import warnings

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='templates')


def load_pickled_object(path):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='.*Trying to unpickle estimator.*',
            category=Warning,
        )
        with open(path, 'rb') as f:
            return pickle.load(f)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'standard.pkl')

model = load_pickled_object(MODEL_PATH)
scaler = load_pickled_object(SCALER_PATH)

TYPE_NAMES = {
    0: 'Grass', 1: 'Fire', 3: 'Water', 4: 'Bug', 5: 'Normal', 6: 'Poison',
    7: 'Electric', 8: 'Ground', 9: 'Fairy', 10: 'Fighting', 11: 'Psychic',
    12: 'Rock', 13: 'Ghost', 14: 'Ice', 15: 'Dragon', 16: 'Dark', 17: 'Steel', 18: 'Flying'
}

FEATURES = ['height_dm', 'weight_hg', 'base_experience', 'hp', 'attack', 'defense']

def predict_type(values):
    feature_data = {feature: float(values.get(feature, 0)) for feature in FEATURES}
    df = pd.DataFrame([feature_data], columns=FEATURES)
    scaled = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled, columns=FEATURES)
    pred = model.predict(scaled_df)[0]
    return TYPE_NAMES.get(int(pred), str(int(pred)))


@app.route('/')
def index():
    return render_template('index.html', fields=FEATURES, prediction=None, error='')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        prediction = predict_type(request.form)
        return render_template(
            'index.html',
            fields=FEATURES,
            prediction=prediction,
            error=''
        )
    except Exception as exc:
        return render_template(
            'index.html',
            fields=FEATURES,
            prediction=None,
            error=f'Prediction failed: {exc}'
        ), 400


if __name__ == '__main__':
    app.run(debug=True)
