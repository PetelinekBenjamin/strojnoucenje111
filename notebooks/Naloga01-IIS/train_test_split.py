import pandas as pd

# Preberi CSV datoteko
data = pd.read_csv(data/processed/reference_data.csv')

# Določi število vrstic za testno datoteko
test_size = int(len(data) * 0.1)

# Razdeli podatke na train in test
train_data = data.iloc[:-test_size]
test_data = data.iloc[-test_size:]

# Shranitev v datoteko train.csv in test.csv
train_data.to_csv('data/processed/train_prod.csv', index=False)
test_data.to_csv('data/processed/test_prod.csv', index=False)
