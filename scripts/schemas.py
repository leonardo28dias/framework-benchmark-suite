"""
Pydantic models for the collection blueprint YAML files.

Configuration flows strictly as:

    YAML file
      -> raw mapping (yaml.safe_load)
      -> CollectionBlueprint (validated configuration)
      -> pipeline

Invalid configurations fail with a clear field-level error before any
expensive work (OpenML search, download, cleaning) starts. The models are
deliberately small: they cover the comparison conditions the pipeline
understands and the keys the definition stage writes, nothing more.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ALLOWED_CONDITION_OPS = frozenset({"==", "!=", "<", "<=", ">", ">="})

# Fields whose condition values must be numeric / boolean, mirroring the query
# language accepted by define_collection.py.
NUMERIC_CONDITION_FIELDS = frozenset({
    "n_instances", "n_features", "n_classes", "missing_rate", "imbalance_ratio", "p_to_n_ratio",
})
BOOLEAN_CONDITION_FIELDS = frozenset({"temporal", "grouped", "academic_usage", "iid_assumed"})

# Fields that only make sense as a rate between 0 and 1.
RATE_CONDITION_FIELDS = frozenset({"missing_rate"})

_COLLECTION_ID_PATTERN = r"^[\w][\w\-]*$"


class ComparisonCondition(BaseModel):
    """One comparison such as ``n_instances >= 500``."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    op: str
    value: Any

    @field_validator("op")
    @classmethod
    def _operator_must_be_supported(cls, value: str) -> str:
        if value not in ALLOWED_CONDITION_OPS:
            raise ValueError(f"unsupported operator {value!r}; expected one of {sorted(ALLOWED_CONDITION_OPS)}")
        return value

    @model_validator(mode="after")
    def _value_must_match_field(self) -> ComparisonCondition:
        if self.field in NUMERIC_CONDITION_FIELDS:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise ValueError(f"field {self.field!r} expects a numeric value, got {self.value!r}")
            if self.field in RATE_CONDITION_FIELDS and not 0 <= float(self.value) <= 1:
                raise ValueError(f"field {self.field!r} must be between 0 and 1, got {self.value!r}")
        elif self.field in BOOLEAN_CONDITION_FIELDS:
            if not isinstance(self.value, bool):
                raise ValueError(f"field {self.field!r} expects a boolean value, got {self.value!r}")
        return self


class OrConditionGroup(BaseModel):
    """One disjunction produced by ``OR`` queries, e.g. ``a OR b``."""

    model_config = ConfigDict(extra="forbid")

    any_of: list[list[ComparisonCondition]] = Field(min_length=2)


AllOfCondition = ComparisonCondition | OrConditionGroup


class SelectionCriteria(BaseModel):
    """The ``selection_criteria`` mapping of a collection blueprint."""

    model_config = ConfigDict(extra="forbid")

    all_of: list[AllOfCondition] = Field(min_length=1)


class SortRule(BaseModel):
    """One entry of ``preferred_sorting``."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    order: Literal["asc", "desc"]


class BlueprintCuration(BaseModel):
    """The ``curation`` block written by the definition stage."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1.0"
    author: str = ""
    date: str = ""


class CollectionBlueprint(BaseModel):
    """Validated schema of a ``collections/<collection_id>.yaml`` file.

    This is the contract between the definition stage (which writes it) and the
    ingestion stage (which reads it): required fields are enforced, types and
    operators are checked, and path-unsafe collection ids are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    collection_id: str = Field(min_length=1, pattern=_COLLECTION_ID_PATTERN)
    name: str = Field(min_length=1)
    description: str = ""
    selection_expression: str | None = None
    selection_criteria: SelectionCriteria
    preferred_sorting: list[SortRule] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)
    curation: BlueprintCuration = Field(default_factory=BlueprintCuration)
