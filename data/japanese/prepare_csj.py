"""Prepare the CSJ dataset"""
import argparse
from collections import namedtuple
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import torchaudio
from tqdm import tqdm

from plearning.utils import df_to_item, read_item

Speaker = namedtuple("Speaker", ["speaker_id", "birth_generation", "gender"])
Segment = namedtuple("Segment", ["seg_id", "speaker_id", "talk_id", "start", "end", "channel"])

GENDERS = {"女": "F", "男": "M"}


class XMLData:
    def __init__(self, root: Path) -> None:
        self.root = root
        segments, speakers = [], {}
        for xml in tqdm(list(self.root.iterdir()), desc="Parsing XML files"):
            talk = ElementTree.parse(xml).getroot()
            talkid = talk.attrib.pop("TalkID")
            if talkid != xml.stem:
                raise ValueError(f"In XML {xml}, talkid {talkid} is invalid.")
            spkid = talk.attrib.pop("SpeakerID")
            if spkid not in speakers:
                speakers[spkid] = Speaker(
                    spkid, talk.attrib["SpeakerBirthGeneration"], GENDERS[talk.attrib["SpeakerSex"]]
                )
            for ipu in talk.findall("IPU"):
                idx = f"{spkid}-{talkid}-{ipu.attrib['IPUID']}"
                start, end = ipu.attrib["IPUStartTime"], ipu.attrib["IPUEndTime"]
                channel = ipu.attrib["Channel"]
                segments.append(Segment(idx, spkid, talkid, start, end, channel))
        self.segments = pd.DataFrame(segments)
        self.speakers = pd.DataFrame(speakers.values())

    def fix_segments(self, wavs: Path) -> None:
        """Fix segments that have incompatible `start` and `end` fields"""
        remove_segment = []
        trim_segment = {}
        for talk_id, subdf in tqdm(list(self.segments.groupby("talk_id")), desc="Fixing segments"):
            file_info = torchaudio.info(wavs / f"{talk_id}.wav")
            duration = file_info.num_frames / file_info.sample_rate
            for idx, row in subdf.iterrows():
                if float(row["start"]) >= duration:
                    remove_segment.append(idx)
                elif float(row["end"]) > duration:
                    trim_segment[idx] = duration
        fixed_segments = self.segments.drop(remove_segment)
        idx, durations = zip(*trim_segment.items())
        fixed_segments.loc[idx, "end"] = durations
        self.segments = fixed_segments

    def write(self, path: Path) -> None:
        self.segments.to_csv(path / "all_segments_xml.csv", index=False)
        self.speakers.to_csv(path / "speakers.csv", index=False)


def talks_from_segments(segments: pd.DataFrame) -> pd.Series:
    """Get unique talks from segments DataFrame"""
    return segments[["talk_id", "speaker_id"]].drop_duplicates()


def patch_item(item: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Patch .item with the correct segment and talk names"""
    item["talk_id"] = item["seg_id"].apply(lambda x: x.split("_")[1])
    talks = talks_from_segments(segments)
    correct_item = item.merge(talks, on="talk_id", suffixes=("_old", ""))
    correct_item["seg_id"] = correct_item.apply(
        lambda row: f"{row.speaker_id}-{row.seg_id.split('_')[1]}-" + f"{row.seg_id.split('_')[2]}", axis=1
    )
    return correct_item.drop(["talk_id", "speaker_id_old"], axis=1)


def patch_test_segments(test: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Format the test segments using information from the reference segments"""
    correct = test.rename({0: "seg_id", 1: "talk_id", 2: "start", 3: "end"}, axis=1)
    correct["channel"] = "L"
    correct["talk_id"] = correct["seg_id"].apply(lambda x: x.split("_")[1])
    talks = talks_from_segments(segments)
    correct = correct.merge(talks, on="talk_id")
    correct["seg_id"] = correct.apply(
        lambda row: f"{row.speaker_id}-{row.seg_id.split('_')[1]}-" + f"{row.seg_id.split('_')[2]}",
        axis=1,
    )
    return correct.merge(
        segments,
        on=segments.columns.difference(["start", "end"]).tolist(),
        suffixes=["_old", ""],
    )[["seg_id", "speaker_id", "talk_id", "start", "end", "channel"]]


def segments_from_pyannote(xml_segments: pd.DataFrame, rttm_dir: Path) -> pd.DataFrame:
    """Convert a rttm file generated with pyannote.audio into a segments DataFrame"""
    rttm_segments = []

    for path in tqdm(list(Path(rttm_dir).glob("*.rttm"))):
        talk_id = path.stem
        sub_segments = xml_segments[xml_segments.talk_id == talk_id]

        if talk_id.startswith("D") or sub_segments.empty:
            continue
        rttm = pd.read_csv(path, sep=" ", header=None)[[1, 3, 4]]
        rttm.rename({1: "talk_id", 3: "start", 4: "duration"}, axis=1, inplace=True)
        rttm["end"] = rttm["start"] + rttm["duration"]
        rttm["talk_id"] = talk_id

        for attribute in ["channel", "speaker_id"]:
            assert len(sub_segments[attribute].unique()) == 1, path
            rttm[attribute] = sub_segments[attribute].iloc[0]

        rttm["seg_id"] = rttm.apply(lambda row: f"{row.speaker_id}-{row.talk_id}-{row.name:04d}", axis=1)
        rttm_segments.append(rttm)

    return pd.concat(rttm_segments)[["seg_id", "speaker_id", "talk_id", "start", "end", "channel"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare the CSJ train and test datasets",
        epilog="[1]: Early phonetic learning without phonetic categories, Schatz et al., PNAS 2021",
    )
    parser.add_argument("dataset", type=str, help="path to the target dataset")
    parser.add_argument("segments_txt", type=str, help="segments.txt file from [1]")
    parser.add_argument("test_item", type=str, help="test.item file from [1]")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    assert (dataset / "raw/wav").is_dir() and (dataset / "raw/wav").is_symlink()
    assert (dataset / "raw/xml").is_dir() and (dataset / "raw/xml").is_symlink()
    assert (dataset / "rttm").is_dir()
    (dataset / "csv").mkdir(exist_ok=True)
    (dataset / "test").mkdir(exist_ok=True)

    xml = XMLData(dataset / "raw/xml")
    xml.fix_segments(dataset / "raw/wav")
    correct_item = patch_item(read_item(args.test_item), xml.segments)
    test_speakers = set(correct_item.speaker_id)
    xml_train_segments = xml.segments[~xml.segments.speaker_id.isin(test_speakers)]
    test_segments = patch_test_segments(pd.read_csv(args.segments_txt, sep=" ", header=None), xml.segments)
    train_segments = segments_from_pyannote(xml_train_segments, dataset / "rttm")

    df_to_item(correct_item, dataset / "test/test.item")
    xml.write(dataset / "csv")
    xml_train_segments.to_csv(dataset / "csv/train_segments_xml.csv", index=False)
    test_segments.to_csv(dataset / "csv/test_segments.csv", index=False)
    train_segments.to_csv(dataset / "csv/train_segments.csv", index=False)
