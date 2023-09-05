import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plearning.phone_pair import PhonePair

DATASETS_LANG = {"csj": "Japanese", "gpj": "Japanese", "wsj": "English", "buc": "English"}


def get_logger(
    name: str, *, filename: Path | None = None, formatter: str = "%(asctime)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    logger = logging.getLogger(name)
    handler: logging.Handler
    if filename is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(filename)
    handler.setFormatter(logging.Formatter(formatter))
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def query(df: pd.DataFrame, **kwargs: Any) -> np.ndarray:
    conditions = []
    for key, value in kwargs.items():
        if value is None or (isinstance(value, PhonePair) and value.first is None):
            conditions.append(df[key].isnull())
        else:
            conditions.append(df[key] == value)
    return np.logical_and.reduce(conditions)


def native_nonnative_selection(
    df: pd.DataFrame, datasets: list[str], **kwargs: Any
) -> tuple[pd.DataFrame, pd.DataFrame]:
    native_idx, non_native_idx = None, None
    for test in datasets:
        this_test = (df["test"] == test) & (df["train"].isin(["English", "Japanese"]))
        is_native = df.train == DATASETS_LANG[test]
        if native_idx is None:
            native_idx = this_test & is_native
            non_native_idx = this_test & (~is_native)
        else:
            native_idx = native_idx | (this_test & is_native)
            non_native_idx = non_native_idx | (this_test & (~is_native))
    return df[native_idx & query(df, **kwargs)], df[non_native_idx & query(df, **kwargs)]


def _is_valid_split(split: int, num: int) -> bool:
    match split:
        case 1:
            return num == 1
        case 5:
            return num == 5
        case 25 | 125 | 500:
            return num == 15
        case _:
            raise ValueError(f"Invalid split {split}: got {num} items.")


def make_pairwise_score(score: pd.DataFrame, col: str) -> pd.DataFrame:
    subdfs = []
    for split, df in score.groupby("split"):
        english = df[df.train == "English"]
        japanese = df[df.train == "Japanese"]
        if len(japanese) == 0:
            subdf = english[col].reset_index(drop=True)
        elif len(english) == 0:
            subdf = japanese[col].reset_index(drop=True)
        else:
            assert english.shape == japanese.shape
            subdf = (english[col].reset_index(drop=True) + japanese[col].reset_index(drop=True)) / 2
        assert _is_valid_split(split, len(subdf)), (split, len(subdf))
        subdf = subdf.to_frame()
        subdf["split"] = split
        subdfs.append(subdf)
    return pd.concat(subdfs, ignore_index=True)


def read_item(filepath_or_buffer: str | Path) -> pd.DataFrame:
    """Read an .item file into a pandas DataFrame"""
    item_file = pd.read_csv(filepath_or_buffer, sep=" ")
    item_file.rename(
        {
            "#file": "seg_id",
            "speaker": "speaker_id",
            "onset": "start",
            "offset": "end",
        },
        axis=1,
        inplace=True,
    )
    return item_file


def df_to_item(dataframe: pd.DataFrame, path_or_buf: str | Path) -> None:
    """Convert a segments DataFrame into a .item file"""
    item = dataframe.rename(
        {
            "seg_id": "#file",
            "start": "onset",
            "end": "offset",
            "speaker_id": "speaker",
        },
        axis=1,
    )[["#file", "onset", "offset", "#phone", "prev-phone", "next-phone", "speaker"]]
    item.to_csv(path_or_buf, sep=" ", index=False)
