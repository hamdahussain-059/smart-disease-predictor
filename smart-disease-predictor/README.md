🦠 COVID-19 Prediction System

A Machine Learning + Flask Web Application

This project implements a machine-learning–based system to predict whether a person is COVID-19 Positive or Negative based on symptoms and exposure-related factors.
It demonstrates the complete ML lifecycle — training, saving, and deploying a model through a simple web interface.

📌 1. Project Overview

The goal of this project is to build an end-to-end ML system:

Training: Preprocess the dataset and train a Random Forest Classifier.

Model Persistence: Save the trained model using pickle (model.pkl).

Deployment: Load and use the model inside a Flask web app.

Real-Time Inference: Users input symptoms, and the model predicts COVID-19 status instantly.

📂 2. File Structure
project/
│── app.py                # Flask backend for prediction
│── model.pkl             # Serialized Random Forest model
│── SDP.ipynb             # Notebook: preprocessing + training + model saving
│── Covid Dataset.csv     # Raw dataset (Yes/No format)
│── Documentation.docx    # Full project documentation
│
└── templates/
     └── index.html       # Frontend form for user inputs


Description of Key Files

File / Folder	Type	Description
app.py	Python Script	Handles routes, loads model.pkl, processes form data, returns prediction.
SDP.ipynb	Jupyter Notebook	Data loading, Yes/No → 1/0 encoding, model training, evaluation, and serialization.
model.pkl	Model File	Trained Random Forest Classifier saved using pickle.
index.html	HTML	UI form for 20 symptoms/exposure feature inputs.
Covid Dataset.csv	Dataset	Binary (Yes/No) dataset with 20 feature columns + target column.
⚙️ 3. Setup & Installation
Prerequisites

Python 3.x

Required Libraries
Flask
pandas
scikit-learn
pickle (built-in)

Installation Steps

Clone the repository

git clone <repository_url>
cd <project_directory>


Install dependencies

pip install pandas scikit-learn Flask


Make sure the following files are present:

app.py
model.pkl
SDP.ipynb
Covid Dataset.csv
templates/index.html

Note: Running the notebook (SDP.ipynb) generates the model.pkl file.

🚀 4. Running the Application

Start the Flask app:

python app.py


Then open the browser and go to:

http://127.0.0.1:5000/


You will see the COVID-19 Prediction Form.

🧪 5. Usage Instructions

Open the web application.

You will see 20 features such as:

Fever

Dry Cough

Asthma

Abroad Travel

Contact with COVID patient

etc.

Select Yes / No for each feature.

Click Predict.

The app will display:

Predicted Status: Positive


or

Predicted Status: Negative

🔬 6. Model Training & Retraining

Training steps implemented in SDP.ipynb:

✔ Data Loading

Load Covid Dataset.csv.

✔ Encoding

Convert all "Yes" → 1, and "No" → 0.

✔ Training

Train a RandomForestClassifier.

✔ Model Saving

Serialize the model with:

pickle.dump(model, open("model.pkl", "wb"))

To retrain:

Simply reopen the notebook and run all cells. The model will automatically overwrite model.pkl.