

import sys

from great_expectations.checkpoint.types.checkpoint_result import (
    CheckpointResult,
)
from great_expectations.data_context import FileDataContext, get_context

data_context: FileDataContext = get_context(
    context_root_dir=r"C:\Users\benja\Desktop\Strojno ucenje\strojnoucenje111-master\strojnoucenje111-master\gx"
)

result: CheckpointResult = data_context.run_checkpoint(
    checkpoint_name="nov_data_checkpoint",
    batch_request=None,
    run_name=None,
)

if not result["success"]:
    print("Validation failed!")
    sys.exit(1)

print("Validation succeeded!")
sys.exit(0)
