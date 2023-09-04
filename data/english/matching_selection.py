import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from plearning.data import symlink_data


def select_matching_subset(csj_train_segments: pd.DataFrame, librivox_segments: pd.DataFrame) -> pd.DataFrame:
    """Select a subset of LibriVox matched to CSJ."""
    csj_train_segments["duration"] = csj_train_segments["end"] - csj_train_segments["start"]
    librivox_segments["duration"] = librivox_segments["end"]
    durations = csj_train_segments.groupby("speaker_id").duration.sum().sort_values()
    durations_librivox = librivox_segments.groupby(["speaker_id"]).duration.sum().sort_values()

    speaker_mapping: dict[str, str] = {}
    duration_difference: dict[str, float] = {}
    librivox_train_segments = []
    for speaker_id, duration in tqdm(durations.items(), total=len(durations)):
        speaker_id_librivox = abs(durations_librivox.drop(speaker_mapping.values()) - duration).idxmin()
        speaker_mapping[speaker_id] = speaker_id_librivox
        diff = durations_librivox[speaker_id_librivox] - duration
        selection = librivox_segments[librivox_segments.speaker_id == speaker_id_librivox].sort_values(by="duration")
        to_remove, removed_duration = [], 0
        for idx, row in selection.iterrows():
            if row.duration + removed_duration > diff:
                break
            to_remove.append(idx)
            removed_duration += row.duration
        to_keep = selection.drop(to_remove)
        librivox_train_segments.append(to_keep)
        duration_difference[speaker_id] = to_keep.duration.sum() - duration

    return pd.concat(librivox_train_segments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select a subset of LibriVox matching the CSJ training set in number"
        + "of speakers and speech quantity per speaker.",
        epilog="Download and segment data with Libri-Light before: https://github.com/facebookresearch/libri-light",
    )
    parser.add_argument("root", help="dataset to build")
    parser.add_argument("csj_train_segments", help="CSV file of Japanese train segments")
    parser.add_argument("librivox_data", help="directory of audio files of LibriVox")
    parser.add_argument("librivox_segments", help="CSV file of all segments in LibriVox")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    csv = root / "csv/train_segments.csv"
    csv.parent.mkdir(exist_ok=True, parents=True)
    csj_train_segments = pd.read_csv(args.csj_train_segments)
    librivox_segments = pd.read_csv(args.librivox_segments)
    train_segments = select_matching_subset(csj_train_segments, librivox_segments)
    train_segments.to_csv(csv)

    symlink_data(train_segments, args.librivox_data, root / "train/full")
