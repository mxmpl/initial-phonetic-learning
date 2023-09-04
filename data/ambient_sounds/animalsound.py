import argparse
import dataclasses
import enum
import warnings
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


class GBIF(enum.StrEnum):
    INDEX = "gbifID"
    URL_KEY = "identifier"
    MULTIMEDIA = "multimedia.txt"
    VERBATIM = "verbatim.txt"
    OCCURRENCE = "occurrence.txt"


@dataclasses.dataclass
class Downloader:
    root: Path

    def __call__(self, entry: pd.Series) -> bool:
        response = requests.get(entry[GBIF.URL_KEY], verify=False)
        if not response.ok:
            return False
        suffix = Path(response.url).suffix
        with open(self.root / f"{entry.name}{suffix}", "wb") as audio_file:
            audio_file.write(response.content)
        return True


def main(root: Path, gbif: Path) -> None:
    warnings.filterwarnings("ignore")
    root.mkdir(exist_ok=True)
    (root / "audio").mkdir(exist_ok=True)

    multimedia = pd.read_csv(gbif / GBIF.MULTIMEDIA, sep="\t").dropna(axis=1, how="all")
    verbatim = pd.read_csv(gbif / GBIF.VERBATIM, sep="\t").dropna(axis=1, how="all")
    occurrence = pd.read_csv(gbif / GBIF.OCCURRENCE, sep="\t").dropna(axis=1, how="all")
    data = multimedia.merge(verbatim, on=GBIF.INDEX).merge(occurrence, on=GBIF.INDEX, suffixes=("", "_occurrence"))
    data.drop(data[data[GBIF.URL_KEY].isna()].index, inplace=True)
    data.set_index(GBIF.INDEX, inplace=True)

    downloader, failed = Downloader(root / "audio"), []
    for idx, entry in tqdm(data.iterrows(), total=data.shape[0], desc="Downloading audio files"):
        if not downloader(entry):
            failed.append(idx)
    data.drop(failed, inplace=True)
    data.to_csv(root / "metadata.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download audio files from the Animal Sound dataset.",
        epilog="Need first to download the raw metadata from https://doi.org/10.15468/dl.dmckt3, CC BY 4.0.",
    )
    parser.add_argument("root", type=str, help="where to build the dataset")
    parser.add_argument("gbif", type=str, help="path to the directory of raw metadata from GBIF")

    args = parser.parse_args()
    main(Path(args.root).resolve(), Path(args.gbif).resolve())
