"""Split training sets"""
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SPLIT_FACTORS: list[int] = [5, 5, 5, 4]  # Split in 5, then 25, 125, and 500


class GroupbyKey(StrEnum):
    SEGMENT = "seg_id"
    FILE = "talk_id"
    SPEAKER = "speaker_id"


def greedy_split(all_segments: list[pd.DataFrame], split: int, groupby_key: GroupbyKey) -> list[pd.DataFrame]:
    """Simple split by a greedy algorithm"""
    result = []
    for segments in all_segments:
        indices: list[list[int]] = [[] for _ in range(split)]
        durations = np.zeros(split)
        duration_by_key = segments.groupby(groupby_key)["duration"].sum()
        for idx, duration in duration_by_key.items():
            min_idx = np.argmin(durations)
            indices[min_idx].append(idx)
            durations[min_idx] += duration
        for idx in indices:
            current_keys = duration_by_key.loc[idx].index
            result.append(segments[segments[groupby_key].isin(current_keys)])
    return result


def symlink_data(
    segments: pd.DataFrame, input_dir: Path, destination: Path, groupby_key: GroupbyKey = GroupbyKey.SEGMENT
) -> pd.DataFrame:
    """Symlink file in partitions to the true files"""
    destination.mkdir()
    for speaker_id, subdf in segments.groupby("speaker_id"):
        if groupby_key == GroupbyKey.SPEAKER:
            (destination / str(speaker_id)).symlink_to(input_dir / str(speaker_id), True)
        elif groupby_key == GroupbyKey.SEGMENT:
            (destination / str(speaker_id)).mkdir()
            for _, row in subdf.iterrows():
                file = f"{row['seg_id']}.wav"
                (destination / f"{speaker_id}/{file}").symlink_to(input_dir / f"{speaker_id}/{file}")
        else:
            raise NotImplementedError(f"Symlink not implemented for {groupby_key}")


def create_partitions(
    full_dir: Path,
    segment_csv: Path,
    groupby_key: GroupbyKey = GroupbyKey.SEGMENT,
    do_symlink: bool = True,
    seed: int = 0,
) -> None:
    """Split the training set into subsets of approximately the same length"""
    full_dir = full_dir.resolve()
    output_dir, csv_dir = full_dir.parent / groupby_key, segment_csv.parent / groupby_key
    output_dir.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)

    segments = pd.read_csv(segment_csv)
    segments["duration"] = segments["end"] - segments["start"]
    partitions = [segments.sample(frac=1, random_state=seed)]

    with tqdm(total=len(SPLIT_FACTORS)) as pbar:
        split = 1
        for split_factor in SPLIT_FACTORS:
            split *= split_factor
            pbar.set_description(f"Split in {split} by {groupby_key}.")
            partitions = greedy_split(partitions, split_factor, groupby_key)
            (output_dir / str(split)).mkdir(exist_ok=True)
            for idx, partition in enumerate(partitions):
                partition["split_id"] = idx
                if do_symlink:
                    symlink_data(partition, full_dir, output_dir / f"{split}/{idx}", groupby_key)
            pd.concat(partitions).to_csv(csv_dir / f"{split}.csv", index=False)
            pbar.update()
