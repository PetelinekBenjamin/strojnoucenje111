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




pot_do_datoteke = 'C:/Users/benja/Desktop/Stuff/Sola/Strojno ucenje2/data/processed/GOSPOSVETSKA C - TURNERJEVA UL.csv'


df = pd.read_csv(pot_do_datoteke, parse_dates=['last_update'], index_col='last_update')


# manjkajoče
print(df.isnull().sum())

# Sortiranje zapisov glede na čas
df.sort_index(inplace=True)

# Izris grafa
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['available_bike_stands'], label='available_bike_stands')
plt.xlabel('Datum')
plt.ylabel('available_bike_stands')
plt.legend()
plt.show()


df_train = df.dropna()

# Ločitev atributov in ciljne spremenljivke
X_train = df_train.drop(['temperature', 'precipitation_probability', 'rain'], axis=1)
y_train_temp = df_train['temperature']
y_train_PP = df_train['precipitation_probability']
y_train_rain = df_train['rain']


rf_model_temp = RandomForestRegressor()
rf_model_PP = RandomForestRegressor()
rf_model_rain = RandomForestRegressor()

# Učenje
rf_model_temp.fit(X_train, y_train_temp)
rf_model_PP.fit(X_train, y_train_PP)
rf_model_rain.fit(X_train, y_train_rain)

# Manjkajoči X
X_missing = df[df.isnull().any(axis=1)].drop(['temperature', 'precipitation_probability', 'rain'], axis=1)

# Napovedovanje
predictions_temp = rf_model_temp.predict(X_missing)
predictions_PP = rf_model_PP.predict(X_missing)
predictions_rain = rf_model_rain.predict(X_missing)


predictions_temp = rf_model_temp.predict(X_missing)
predictions_PP = rf_model_PP.predict(X_missing)
predictions_rain = rf_model_rain.predict(X_missing)

# Dodelitev
df.loc[df['temperature'].isnull(), 'temperature'] = predictions_temp[:len(df.loc[df['temperature'].isnull()])]
df.loc[df['precipitation_probability'].isnull(), 'precipitation_probability'] = predictions_PP[:len(df.loc[df['precipitation_probability'].isnull()])]
df.loc[df['rain'].isnull(), 'rain'] = predictions_rain[:len(df.loc[df['rain'].isnull()])]

# Preverjanje
print(df.isnull().sum())

# datumski objekti
datum = pd.to_datetime(df.index, format='%d/%m/%Y')

# Dodajanje stolpcev day, month in year
df['day'] = datum.day
df['month'] = datum.month
df['year'] = datum.year


print(df.columns)
print(df.tail())

doprinos, _ = f_regression(df.drop(['available_bike_stands'], axis=1), df['available_bike_stands'])


for zanc, dop in zip(df.columns,doprinos):
    print(f'{zanc} ima doprinos: {dop}')


# Filtriranje značilnic
najdoprinosne_znacilnice = ['temperature', 'relative_humidity', 'apparent_temperature', 'dew_point']
ciljna_znacilnica = 'available_bike_stands'
podatki = df[najdoprinosne_znacilnice + [ciljna_znacilnica]]


scaler = StandardScaler()
podatki_standardized = scaler.fit_transform(podatki[['temperature', 'relative_humidity', 'apparent_temperature', 'dew_point']])


pot_do_scalerja = "C:/Users/benja/Desktop/Stuff/Sola/Strojno ucenje2/models/naloga01_scalerTest1.pkl"
joblib.dump(scaler, pot_do_scalerja)



scaler1 = StandardScaler()
podatki_standardized1 = scaler1.fit_transform(podatki[['available_bike_stands']])


pot_do_scalerja1 = "C:/Users/benja/Desktop/Stuff/Sola/Strojno ucenje2/models/naloga01_scalerTest2.pkl"
joblib.dump(scaler1, pot_do_scalerja1)


podatki_standardized = podatki_standardized + podatki_standardized1


# Ločitev na učno in testno množico
train_size = len(podatki) - 1500
train_data, test_data = podatki_standardized[:train_size], podatki_standardized[train_size:]

# Preverjanje oblike podatkov
print("Oblika učnih podatkov:", train_data.shape)
print("Oblika testnih podatkov:", test_data.shape)
print()

def pripravi_podatke_za_ucenje(vrednosti, okno_velikost):
    X, y = [], []
    for i in range(len(vrednosti) - okno_velikost):
        X.append(vrednosti[i:i+okno_velikost, :])
        y.append(vrednosti[i+okno_velikost, -1])
    return np.array(X), np.array(y)


# velikost okna
okno_velikost = 60

# Priprava učnih podatkov
X_train, y_train = pripravi_podatke_za_ucenje(train_data, okno_velikost)

# Priprava testnih podatkov
X_test, y_test = pripravi_podatke_za_ucenje(test_data, okno_velikost)

# Izbris dimenzije
y_train = y_train.flatten()
y_test = y_test.flatten()

# Definicija modela
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
lstm1 = LSTM(256, return_sequences=True)(inputs)
lstm2 = LSTM(256)(lstm1)
dense1 = Dense(64, activation='relu')(lstm2)
dropout = Dropout(0.2)(dense1)
outputs = Dense(1)(dropout)

# Definicija modela
model_lstm = Model(inputs=inputs, outputs=outputs)

# Kompilacija modela
model_lstm.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# učenja modela
history = model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=1)


# Shranjevanje modela
model_lstm.save("C:/Users/benja/Desktop/Stuff/Sola/Strojno ucenje2/models/naloga01_modelTest.h5")




# Preverjanje uspešnosti modela
y_pred = model_lstm.predict(X_test)



print("Oblika y_pred:", y_pred.shape)
print("Oblika y_test:", y_test.shape)



mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Data:", mse)


mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error on Test Data:", mae)


r2 = r2_score(y_test, y_pred)
print("R^2 on Test Data:", r2)



y_pred_unscaled = scaler1.inverse_transform(y_pred.reshape(-1, 1)).flatten()

print("unscaled: ", y_pred_unscaled)


# Zbiranje metrik nad učnimi podatki
train_metrics = {
    'MSE': mean_squared_error(y_train, model_lstm.predict(X_train)),
    'MAE': mean_absolute_error(y_train, model_lstm.predict(X_train)),
    'R^2': r2_score(y_train, model_lstm.predict(X_train))
}

# Shranjevanje metrik nad učnimi podatki
with open(r"C:/Users/benja/Desktop/Stuff/Sola/Strojno ucenje2/reports/train_metrics.txt", 'w') as f:
    for metric, value in train_metrics.items():
        f.write(f'{metric}: {value}\n')


test_metrics = {
    'MSE': mse,
    'MAE': mae,
    'R^2': r2
}

# Shranjevanje metrik nad testnimi podatki v datoteko
with open(r"C:/Users/benja/Desktop/Stuff/Sola/Strojno ucenje2/reports/metrics.txt", 'w') as f:
    for metric, value in test_metrics.items():
        f.write(f'{metric}: {value}\n')
