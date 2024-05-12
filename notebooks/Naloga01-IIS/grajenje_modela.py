import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import mlflow
from sklearn.base import TransformerMixin
import os
from sklearn.preprocessing import FunctionTransformer

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

def fill_missing_values(X):
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return X_imputed



# Začetek MLflow teka
with mlflow.start_run(run_name="MyModelTrainingUcenje"):
    # Preberi podatke
    pot_do_datoteke = r'data/processed/train_prod.csv'
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
        ('scaler', StandardScaler(), ['temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 'dew_point_2m', 'day', 'month', 'year']),
        ('scaler1', StandardScaler(), ['available_bike_stands']),
    ])




# Uporaba cevovoda za predprocesiranje podatkov
    processed_data = preprocessing_pipeline.fit_transform(podatki)

    print(processed_data.shape)

    train_data = processed_data

    # Pridobitev skalirnika iz cevovoda za predprocesiranje podatkov
    scaler = preprocessing_pipeline.named_transformers_['scaler']

    # Shranjevanje scalera v datoteko
    scaler_path = r"models/scaler_pipeline1.pkl"
    joblib.dump(scaler, scaler_path)





    # Pridobitev skalirnika iz cevovoda za predprocesiranje podatkov
    scaler1 = preprocessing_pipeline.named_transformers_['scaler1']

    # Shranjevanje scalera v datoteko
    scaler_path1 = r"models/scaler_pipeline2.pkl"
    joblib.dump(scaler1, scaler_path1)

    print("Oblika učnih podatkov:", train_data.shape)

    # Priprava podatkov za model
    okno_velikost = 4
    X_train, y_train = pripravi_podatke_za_ucenje(train_data, okno_velikost)
    y_train = y_train.flatten()

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

    # Shranjevanje modela
    model_lstm.save(r"models/model_lstm.h5")



