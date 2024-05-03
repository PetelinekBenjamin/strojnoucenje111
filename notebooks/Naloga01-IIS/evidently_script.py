import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing

from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *

# Preberi podatke iz CSV-ja
file_path = r'data/processed/reference_data.csv'
housing_data = pd.read_csv(file_path)

# Pretvori podatke v DataFrame
housing_data = pd.DataFrame(housing_data)

# Izberi stolpce, ki jih boš uporabil/a za analizo
columns_to_analyze = ['precipitation_probability', 'apparent_temperature', 'dew_point_2m', 'relative_humidity_2m', 'temperature_2m']

# Dodaj naključne napovedi za potrebe primera
for column in columns_to_analyze:
    housing_data[column + '_prediction'] = housing_data[column].values + np.random.normal(0, 5, housing_data.shape[0])

# Izberi vzorce za primerjavo (če želiš)
reference = housing_data.sample(n=50, replace=False)
current = housing_data.sample(n=50, replace=False)

report = Report(metrics=[
    DataDriftPreset(),
])

report.run(reference_data=reference, current_data=current)

# Prikaži poročilo
report.save_html("data/processed/rift_test.html")

tests = TestSuite(tests=[
    TestNumberOfColumnsWithMissingValues(),
    TestNumberOfRowsWithMissingValues(),
    TestNumberOfConstantColumns(),
    TestNumberOfDuplicatedRows(),
    TestNumberOfDuplicatedColumns(),
    TestColumnsType(),
    TestNumberOfDriftedColumns(),
])

tests.run(reference_data=reference, current_data=current)

tests.save_html("data/processed/stability_test.html")
