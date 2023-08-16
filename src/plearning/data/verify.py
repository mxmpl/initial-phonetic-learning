"""Verify data"""
from pathlib import Path

import pandas as pd

from plearning import CPC
from plearning.data.partition import GroupbyKey
from plearning.utils import get_logger


def verify(
    full_dir: Path, csv_dir: Path, groupby_key: GroupbyKey = GroupbyKey.SEGMENT, fraction: float = 1.0, seed: int = 0
) -> None:
    """Verify dataset integrity"""
    logger = get_logger("verify")
    train_segments = pd.read_csv(csv_dir / "train_segments.csv")

    def assert_directory(segments: pd.DataFrame, folder: Path) -> None:
        num_files = len(list(folder.rglob(f"*{CPC.file_extension}")))
        if len(segments) != num_files:
            logger.error(f"Invalid number of files for {folder}: {num_files} instead of {len(segments)}")
            return
        for _, row in segments.sample(frac=fraction, random_state=seed).iterrows():
            current_file = (folder / f"{row.speaker_id}/{row.seg_id}").with_suffix(CPC.file_extension)
            if not current_file.is_file():
                logger.error(f"Missing file: {current_file}")

    logger.info("Checking full")
    assert_directory(train_segments, full_dir)
    assert (csv_dir / groupby_key).is_dir()
    for path in (csv_dir / groupby_key).glob("*.csv"):
        segments = pd.read_csv(path)
        assert len(segments) == len(train_segments)
        splits = int(path.stem)
        logger.info(f"Checking for {splits} splits")
        assert (full_dir.parent / groupby_key / str(splits)).is_dir()
        assert len(segments.split_id.unique()) == splits
        for split, subsegs in segments.groupby(groupby_key, as_index=False):
            logger.info(f"Checking partition of idx {split} of split {splits}")
            assert_directory(subsegs, full_dir.parent / groupby_key / f"{splits}/{split}")
