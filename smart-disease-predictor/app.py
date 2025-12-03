from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Features (must match dataset order)
features = [
    'Breathing Problem',
    'Fever',
    'Dry Cough',
    'Sore throat',
    'Running Nose',
    'Asthma',
    'Chronic Lung Disease',
    'Headache',
    'Heart Disease',
    'Diabetes',
    'Hyper Tension',
    'Fatigue ',
    'Gastrointestinal ',
    'Abroad travel',
    'Contact with COVID Patient',
    'Attended Large Gathering',
    'Visited Public Exposed Places',
    'Family working in Public Exposed Places',
    'Wearing Masks',
    'Sanitization from Market'
]


# Convert Yes/No to 1/0
def encode_value(val):
    return 1 if val == "Yes" else 0

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = []

    for f in features:
        value = request.form.get(f)
        input_data.append(encode_value(value))

    df = pd.DataFrame([input_data], columns=features)

    pred = model.predict(df)[0]

    result = "COVID Positive" if pred == 1 else "COVID Negative"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
