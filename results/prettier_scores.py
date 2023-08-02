import argparse
from collections import defaultdict
from pathlib import Path
from typing import Literal

import pandas as pd

from plearning.phone_pair import PhonePair
from plearning.utils import query

ROOT = Path("./abx/")


def get_jap(mean: pd.DataFrame, std: pd.DataFrame, score: str) -> tuple[float, float]:
    csj = ((1 - mean[mean.test == "csj"][score].iloc[0]) * 100, std[std.test == "csj"][score].iloc[0] * 100)
    gpj = ((1 - mean[mean.test == "gpj"][score].iloc[0]) * 100, std[std.test == "gpj"][score].iloc[0] * 100)
    if std[std.test == "gpj"][score].iloc[0] == 0:
        return f"{csj[0]:0.1f} / {gpj[0]:0.1f}"
    return f"{csj[0]:0.1f} ({csj[1]:0.1f}) / {gpj[0]:0.1f} ({gpj[1]:0.1f})"


def get_english(mean: pd.DataFrame, std: pd.DataFrame, score: str) -> tuple[float, float]:
    buc = ((1 - mean[mean.test == "buc"][score].iloc[0]) * 100, std[std.test == "buc"][score].iloc[0] * 100)
    wsj = ((1 - mean[mean.test == "wsj"][score].iloc[0]) * 100, std[std.test == "wsj"][score].iloc[0] * 100)
    if std[std.test == "buc"][score].iloc[0] == 0:
        return f"{buc[0]:0.1f} / {wsj[0]:0.1f}"
    return f"{buc[0]:0.1f} ({buc[1]:0.1f}) / {wsj[0]:0.1f} ({wsj[1]:0.1f})"


def main(mode: Literal["within", "across"], phone_pair: PhonePair) -> pd.DataFrame:
    no_pretraining = pd.read_csv(ROOT / "no_pretraining.csv")
    noise_pretrained = pd.read_csv(ROOT / "noise_pretraining.csv")
    crossling = pd.read_csv(ROOT / "crossling_pretraining.csv")
    scores = no_pretraining.merge(
        crossling,
        on=["test", "train", "phone_pair", "split", "idx", "mode"],
        suffixes=["", "_crossling"],
        how="outer",
    ).merge(
        noise_pretrained,
        on=["test", "train", "phone_pair", "split", "idx", "mode"],
        suffixes=["_baseline", "_noise"],
        how="outer",
    )
    scores = scores[query(scores, mode=mode, phone_pair=phone_pair)]
    lang_available = scores.test.unique()

    mean = scores.groupby(["test", "train", "split"], as_index=False)[
        ["score_baseline", "score_crossling", "score_noise"]
    ].mean()
    std = scores.groupby(["test", "train", "split"], as_index=False)[
        ["score_baseline", "score_crossling", "score_noise"]
    ].std(ddof=0)

    table = defaultdict(list)
    for col, setup in zip(
        ["score_baseline", "score_crossling", "score_noise"],
        ["No pretraining", "Cross-lingual", "Ambient sounds"],
    ):
        for (train, split), submean in mean.groupby(["train", "split"]):
            substd = std[query(std, train=train, split=split)]
            assert len(substd) == len(submean)
            table["Pretraining"].append(setup)
            table["Training"].append(f"{train} {500//split}h")
            if "csj" in lang_available:
                table["Japanese (CSJ / GPJ)"].append(get_jap(submean, substd, col))
            if "buc" in lang_available:
                table["English (Buckeye / WSJ)"].append(get_english(submean, substd, col))
    return pd.DataFrame(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["within", "across"], type=str)
    parser.add_argument("--phone_pair", type=str)
    args = parser.parse_args()

    if args.phone_pair is None:
        phone_pair = PhonePair(None)
    else:
        assert len(args.phone_pair) == 2
        phone_pair = PhonePair(first_phone=args.phone_pair[0], second_phone=args.phone_pair[1])
    table = main(args.mode, phone_pair)
    print(table.to_markdown(index=False))
