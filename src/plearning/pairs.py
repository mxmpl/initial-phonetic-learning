import itertools
import pickle
from collections import namedtuple
from pathlib import Path

import pandas as pd

Score = namedtuple("Score", ["first_phone", "second_phone", "mode", "score"])


def abx_pairs(root: Path) -> None:
    """Compute ABX errors for every pair for each experiment found"""
    for extras_path in root.resolve().rglob("extras.pkl"):
        if (extras_path.parent / "ABX_pairs.csv").exists():
            continue
        with open(extras_path, "rb") as file:
            extras = pickle.load(file)
        phone_match = extras["phone_match"]

        df = []
        for mode in ["within", "across"]:
            if f"phone_confusion_{mode}" not in extras:
                continue
            phone_confusion = extras[f"phone_confusion_{mode}"]
            divisor_speaker = extras[f"divisor_speaker_{mode}"]
            for first_phone, second_phone in itertools.permutations(phone_match, 2):
                div_spk = divisor_speaker[phone_match[first_phone], phone_match[second_phone]]
                if div_spk == 0:
                    continue
                score = phone_confusion[phone_match[first_phone], phone_match[second_phone]].item()
                df.append(Score(first_phone, second_phone, mode, score))
        pd.DataFrame(df).to_csv(extras_path.parent / "ABX_pairs.csv", index=False)
