import json
from collections import namedtuple
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from plearning import CPC

Score = namedtuple("Score", ["test", "train", "phone_pair", "split", "idx", "mode", "epoch", "score"])


def recap_scores(results: Path, output: Path) -> None:
    """Recap all scores"""
    scores = []

    for path in tqdm(list(results.resolve().rglob("ABX_args.json"))):
        with open(path, "r") as f:
            abx_args = json.load(f)
        checkpoint = Path(abx_args["path_checkpoint"]).resolve()
        mode = abx_args["mode"]
        item_file = abx_args["path_item_file"]
        epoch = int(checkpoint.stem.split("_")[1])
        train = checkpoint.parent.stem.split("_")[0].removeprefix("cpc-")
        test_candidates = [test for test, item in CPC.test_items.items() if str(item) == item_file]
        assert len(test_candidates) == 1
        test = test_candidates[0]
        split_str, idx_str = checkpoint.parent.stem.split("_")[-2:]
        if idx_str == "full":
            split, idx = 1, 0
        else:
            split, idx = int(split_str), int(idx_str)
        with open(path.parent / "ABX_scores.json", "r") as f:
            score = float(json.load(f)[mode])
        scores.append(Score(test, train, None, split, idx, mode, epoch, score))
        path_pairs = path.parent / "ABX_pairs.csv"
        if path_pairs.exists():
            df = pd.read_csv(path_pairs)
            df["phone_pair"] = df.apply(
                lambda row: f"[{row.first_phone}]-[{row.second_phone}]"
                if row.first_phone < row.second_phone
                else f"[{row.second_phone}]-[{row.first_phone}]",
                axis=1,
            )
            for phone_pair, subdf in df.groupby("phone_pair"):
                score = subdf.score.mean()
                scores.append(Score(test, train, phone_pair, split, idx, mode, epoch, score))
        else:
            print(f"Pairs do not exist for {path.parent}")

    pd.DataFrame(scores).to_csv(output, index=False)
