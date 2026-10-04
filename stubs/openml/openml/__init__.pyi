from __future__ import annotations

import openml.datasets  # re-export
import openml.exceptions  # re-export

# Re-assign to help static analysers treat these as public attributes.
datasets = openml.datasets
exceptions = openml.exceptions
