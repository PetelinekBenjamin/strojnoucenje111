import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import mlflow
import os
from tensorflow.keras.models import load_model

# Nastavitev sledenja MLflow
mlflow.set_tracking_uri('https://dagshub.com/PetelinekBenjamin/strojnoucenje111.mlflow')
# mlflow.set_tracking_uri("https://dagshub.com/ZanPovseGit/inteligentniSistem.mlflow")

# Nastavitev uporabniškega imena in gesla
os.environ["MLFLOW_TRACKING_USERNAME"] = "PetelinekBenjamin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "30663408e580bdb3f66e074627577a040f36b5ff"
# os.environ["MLFLOW_TRACKING_USERNAME"] = "ZanPovseGit"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "bdf091cc3f58df2c8346bb8ce616545e0e40b351"

def pripravi_podatke_za_ucenje(vrednosti, okno_velikost):
    X, y = [], []
    for i in range(len(vrednosti) - okno_velikost):
        X.append(vrednosti[i:i+okno_velikost, :])
        y.append(vrednosti[i+okno_velikost, -1])
    return np.array(X), np.array(y)

# Začetek MLflow teka
with mlflow.start_run(run_name="MyModelTrainingOvrednotenje"):


    # Preberi podatke

    pot_do_datoteke = r'data/processed/test_prod.csv'
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
    najdoprinosne_znacilnice = ['temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 'dew_point_2m']
    ciljna_znacilnica = 'available_bike_stands'
    podatki = df[najdoprinosne_znacilnice + [ciljna_znacilnica]]

    # Standardizacija podatkov
    pot_do_scalerja = "models/naloga01_scalerTest3.pkl"
    scaler = joblib.load(pot_do_scalerja)
    podatki_standardized = scaler.fit_transform(podatki[['temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 'dew_point_2m']])




    pot_do_scalerja1 = "models/naloga01_scalerTest4.pkl"
    scaler1 = joblib.load(pot_do_scalerja1)
    podatki_standardized1 = scaler1.fit_transform(podatki[['available_bike_stands']])



    podatki_standardized = podatki_standardized + podatki_standardized1


    test_data = podatki_standardized


    print("Oblika učnih podatkov:", test_data.shape)

    # Priprava podatkov za model
    okno_velikost = 4
    X_test, y_test = pripravi_podatke_za_ucenje(test_data, okno_velikost)
    y_test = y_test.flatten()



    model_lstm = load_model(r"models/model_lstm.h5")



    # Preverjanje uspešnosti modela
    y_pred = model_lstm.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Beleženje metrik
    mlflow.log_metrics({"mse": mse, "mae": mae, "r2": r2})

    print("MSE:", mse)
    print("MAE:", mae)
    print("R2:", r2)
