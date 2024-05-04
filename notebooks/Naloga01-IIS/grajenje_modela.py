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

# Nastavitev sledenja MLflow
mlflow.set_tracking_uri('https://dagshub.com/PetelinekBenjamin/strojnoucenje111.mlflow')

# Nastavitev uporabniškega imena in gesla
os.environ["MLFLOW_TRACKING_USERNAME"] = "PetelinekBenjamin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "30663408e580bdb3f66e074627577a040f36b5ff"

def pripravi_podatke_za_ucenje(vrednosti, okno_velikost):
    X, y = [], []
    for i in range(len(vrednosti) - okno_velikost):
       X.append(vrednosti[i:i+okno_velikost, :])
       y.append(vrednosti[i+okno_velikost, -1])
    return np.array(X), np.array(y)

# Začetek MLflow teka
with mlflow.start_run(run_name="MyModelTraining"):



    # Preberi podatke
    pot_do_datoteke = r'data/processed/train_prod.csv'
    df = pd.read_csv(pot_do_datoteke, parse_dates=['last_update'], index_col='last_update')

    pot_do_datoteke1 = r'data/processed/test_prod.csv'
    df1 = pd.read_csv(pot_do_datoteke1, parse_dates=['last_update'], index_col='last_update')

    df = pd.concat([df, df1])

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
    scaler = StandardScaler()
    podatki_standardized = scaler.fit_transform(podatki[['temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 'dew_point_2m']])
    pot_do_scalerja = "models/naloga01_scalerTest3.pkl"
    joblib.dump(scaler, pot_do_scalerja)

    scaler1 = StandardScaler()
    podatki_standardized1 = scaler1.fit_transform(podatki[['available_bike_stands']])
    pot_do_scalerja1 = "models/naloga01_scalerTest4.pkl"
    joblib.dump(scaler1, pot_do_scalerja1)

    podatki_standardized = podatki_standardized + podatki_standardized1

    # Razdelitev na učne in testne podatke
    test_size = int(len(podatki) * 0.1)
    train_data = podatki_standardized[:-test_size]
    test_data = podatki_standardized[-test_size:]

    print("Oblika učnih podatkov:", train_data.shape)
    print("Oblika testnih podatkov:", test_data.shape)

    # Priprava podatkov za model
    okno_velikost = 4
    X_train, y_train = pripravi_podatke_za_ucenje(train_data, okno_velikost)
    X_test, y_test = pripravi_podatke_za_ucenje(test_data, okno_velikost)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Definicija modela
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    lstm1 = LSTM(256, return_sequences=True)(inputs)
    lstm2 = LSTM(256)(lstm1)
    dense1 = Dense(64, activation='relu')(lstm2)
    dropout = Dropout(0.2)(dense1)
    outputs = Dense(1)(dropout)
    model_lstm = Model(inputs=inputs, outputs=outputs)

    # Kompilacija modela
    model_lstm.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Učenje modela
    history = model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=1)

    # Beleženje parametrov
    mlflow.log_param("epochs", 10)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("validation_split", 0.1)
    mlflow.log_param("patience", 3)

    # Beleženje modela
    mlflow.keras.log_model(model_lstm, "model")

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
