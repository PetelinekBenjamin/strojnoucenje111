import datetime

import pandas as pd

import great_expectations as gx
import great_expectations.jupyter_ux
from great_expectations.core.batch import BatchRequest
from great_expectations.checkpoint import SimpleCheckpoint
from great_expectations.exceptions import DataContextError

context = gx.get_context()

batch_request = {'datasource_name': 'my_datasource', 'data_connector_name': 'default_inferred_data_connector_name', 'data_asset_name': 'GOSPOSVETSKA C - TURNERJEVA UL.csv', 'limit': 1000}

expectation_suite_name = "my_suite"

print(batch_request)


validator = context.get_validator(
    batch_request=BatchRequest(**batch_request),
    expectation_suite_name=expectation_suite_name
)
column_names = [f'"{column_name}"' for column_name in validator.columns()]
print(f"Columns: {', '.join(column_names)}.")
validator.head(n_rows=5, fetch_all=False)

exclude_column_names = [
    "number",
    "contract_name",
    "name",
    "address",
    "position",
    "banking",
    "bonus",
    "bike_stands",
    "available_bike_stands",
    "available_bikes",
    "status",
    "last_update",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation_probability",
]


print(validator.get_expectation_suite(discard_failed_expectations=False))
validator.save_expectation_suite(discard_failed_expectations=False)

checkpoint_config = {
    "class_name": "SimpleCheckpoint",
    "validations": [
        {
            "batch_request": batch_request,
            "expectation_suite_name": expectation_suite_name
        }
    ]
}
checkpoint = SimpleCheckpoint(
    f"{validator.active_batch_definition.data_asset_name}_{expectation_suite_name}",
    context,
    **checkpoint_config
)
checkpoint_result = checkpoint.run()

checkpoint_result = checkpoint.run()

# Preveri, ali je preverjanje uspešno ali ne
success = checkpoint_result["success"]

if success:
    print("Preverjanje uspešno! 🎉")
else:
    print("Preverjanje ni uspešno. 😞")


