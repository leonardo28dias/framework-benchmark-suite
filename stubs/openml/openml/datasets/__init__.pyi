from __future__ import annotations

from typing import Any

import pandas as pd


class OpenMLDataset:
    name: str
    id: int
    licence: str | None
    license: str | None
    citation: str | None
    paper_url: str | None
    default_target_attribute: str | None

    def get_data(
        self,
        target: list[str] | str | None = None,
        include_row_id: bool = False,
        include_ignore_attribute: bool = False,
        dataset_format: str = "dataframe",
    ) -> tuple[Any, Any, list[bool], list[str]]: ...


def get_dataset(
    dataset_id: int | str,
    download_data: bool = False,
    **kwargs: Any,
) -> OpenMLDataset: ...


def list_datasets(
    data_id: list[int] | None = None,
    offset: int | None = None,
    size: int | None = None,
    status: str | None = None,
    tag: str | None = None,
    output_format: str = "dict",
    **kwargs: Any,
) -> dict[str, Any] | pd.DataFrame: ...
