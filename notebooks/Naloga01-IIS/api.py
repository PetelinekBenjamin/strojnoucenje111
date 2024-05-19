import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU

import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from flask import Flask, request, jsonify
import requests
import io
from flask_cors import CORS
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
import tensorflow as tf
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime, timedelta

# MongoDB Atlas connection string
MONGO_URI = "mongodb+srv://benjaminpetelinek12871:user123@cluster0.dk6r56e.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Get the database and collection
db = client.metrike  # Replace with your database name
collection = db.ovrednotenje # Replace with your collection name

# Pot do shranjenega modela
#model_filename = r"C:\Users\benja\Desktop\Strojno_ucenje_Koncna_Verzija\strojnoucenje111-master\models\model_lstm.h5"
model_filename = "models/model_lstm.h5"
# Uvoz modela
with tf.device('/CPU:0'):
    model = load_model(model_filename)

# Pot do shranjenega scalerja
#scaler_path = r"C:\Users\benja\Desktop\Strojno_ucenje_Koncna_Verzija\strojnoucenje111-master\models\scaler_pipeline1.pkl"
scaler_path = "models/scaler_pipeline1.pkl"
# Uvoz scalerja
scaler = joblib.load(scaler_path)

# Pot do shranjenega scalerja
#scaler_path1 = r"C:\Users\benja\Desktop\Strojno_ucenje_Koncna_Verzija\strojnoucenje111-master\models\scaler_pipeline2.pkl"
scaler_path1 = "models/scaler_pipeline2.pkl"
# Uvoz scalerja
scaler1 = joblib.load(scaler_path1)

app = Flask(__name__)
CORS(app)

def pripravi_podatke_za_ucenje(vrednosti, okno_velikost):
    X = []
    for i in range(len(vrednosti) - okno_velikost + 1):
        X.append(vrednosti[i:i+okno_velikost, :])
    return np.array(X)

def fill_missing_values(X):
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return X_imputed

@app.route('/predict/naloga02', methods=['GET'])
def post_example():
    pot_do_datoteke = 'data/processed/test_prod.csv'
    #pot_do_datoteke = r'C:\Users\benja\Desktop\Strojno_ucenje_Koncna_Verzija\strojnoucenje111-master\data\processed\reference_data.csv'
    df = pd.read_csv(pot_do_datoteke, parse_dates=['last_update'], index_col='last_update')

    # Izloči manjkajoče vrednosti
    print(df.isnull().sum())

    # Sortiranje po času
    df.sort_index(inplace=True)

    # Dodajanje stolpcev day, month in year
    datum = pd.to_datetime(df.index, format='%d/%m/%Y')
    df['day'] = datum.day
    df['month'] = datum.month
    df['year'] = datum.year

    print(df.columns)
    print(df.tail())

    # Filtriranje značilnic
    najdoprinosne_znacilnice = ['temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 'dew_point_2m', 'day', 'month', 'year']
    ciljna_znacilnica = 'available_bike_stands'
    podatki = df[najdoprinosne_znacilnice + [ciljna_znacilnica]]

    # Definicija cevovoda za predprocesiranje podatkov
    preprocessing_pipeline = ColumnTransformer([
        ('fill_missing', FunctionTransformer(fill_missing_values), ['temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 'dew_point_2m']),
        ('scaler', scaler, ['temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 'dew_point_2m', 'day', 'month', 'year']),
        ('scaler1', scaler1, ['available_bike_stands']),
    ])

    processed_data = preprocessing_pipeline.fit_transform(podatki)

    test_data = processed_data
    print("Oblika učnih podatkov:", test_data.shape)

    # Priprava podatkov za model
    okno_velikost = 4
    X_test = pripravi_podatke_za_ucenje(test_data, okno_velikost)
    stevilo_podatkov = X_test.shape[0]

    y_pred = []
    for i in range(stevilo_podatkov - 7, stevilo_podatkov):
        with tf.device('/CPU:0'):
            pred = model.predict(X_test[i:i+1])
        y_pred.append(pred)

    y_pred_unscaled = scaler1.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
    rounded_y_pred = [round(num, 1) for num in y_pred_unscaled.tolist()]

    # Current timestamp
    current_time = datetime.now()

    # Insert each prediction separately with its own timestamp incremented by one hour
    for i, prediction in enumerate(rounded_y_pred):
        prediction_time = current_time + timedelta(hours=i+1)
        prediction_data = {
            "timestamp": prediction_time.isoformat(),
            "prediction": prediction
        }
        try:
            collection.insert_one(prediction_data)
            print(f"Prediction data for {prediction_time.isoformat()} inserted into MongoDB successfully.")
        except Exception as e:
            print(f"An error occurred while inserting data into MongoDB: {e}")

    return jsonify({"prediction": rounded_y_pred})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0')
