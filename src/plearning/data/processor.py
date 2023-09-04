import dataclasses
import logging
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Callable

import pandas as pd
from joblib import Parallel, delayed

from plearning import CPC
from plearning.data.partition import GroupbyKey
from plearning.utils import get_logger


class Channels(StrEnum):
    MONO = "mono"
    LEFT = "left"
    RIGHT = "right"
    IGNORE = "ignore"


@dataclasses.dataclass
class SoxProcessor:
    sample_rate: int = 16_000
    extension: str = CPC.file_extension
    channels: Channels = Channels.IGNORE
    precision: int = 16

    def __post_init__(self) -> None:
        assert self.channels in ["left", "right", "mono", "ignore"]
        if not self.extension.startswith(".") or self.extension == ".":
            raise ValueError("extension must start with a dot")
        self._cmd = "sox -G {inp}" + f" -r {self.sample_rate} -b {self.precision}" + " {output}"
        if self.channels == "left":
            self._cmd += " remix 1"
        elif self.channels == "right":
            self._cmd += " remix 2"
        elif self.channels == "mono":
            self._cmd += " remix 1,2"
        self._trim = " trim {start} {duration}"

    def __call__(
        self, inp: str | Path, output: str | Path, start: float | None = None, end: float | None = None
    ) -> tuple[str, str]:
        assert Path(output).suffix == self.extension
        cmd = self._cmd.format(inp=inp, output=output)
        if start is not None and end is not None:
            assert 0 <= start < end
            cmd += self._trim.format(start=start, duration=end - start)
        process = subprocess.run(cmd, capture_output=True, shell=True, text=True)
        if process.returncode != 0:
            raise RuntimeError(process.stderr)
        return process.stdout, process.stderr


def _make_worker(processor: SoxProcessor, logger: logging.Logger) -> Callable:
    def worker(inp: Path, output: Path) -> None:
        stdout, stderr = processor(inp, output)
        if stdout:
            logger.info(f"{inp.name} - stdout: {stdout}")
        if stderr:
            logger.warning(f"{inp.name} - stderr: {stderr}")

    return worker


def process_audio(input_dir: Path, output_dir: Path, channels: Channels = Channels.IGNORE, n_jobs: int = -1) -> None:
    """Process audio files with sox"""
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    output_dir.mkdir(exist_ok=True)
    worker = _make_worker(SoxProcessor(channels=channels), get_logger("sox", filename=output_dir / "process.log"))

    input_files = sorted([file for file in input_dir.glob("*") if file.is_file()])
    output_files = [output_dir / file.with_suffix(CPC.file_extension).name for file in input_files]

    launcher = Parallel(n_jobs=n_jobs, verbose=10)
    launcher(delayed(worker)(inp, output) for inp, output in zip(input_files, output_files))


def create_segments(
    segments_csv: Path, input_dir: Path, output_dir: Path, channels: Channels = Channels.IGNORE, n_jobs: int = -1
) -> None:
    """Create segments with sox"""
    segments = pd.read_csv(segments_csv)
    output_dir.mkdir(exist_ok=True)
    worker = _make_worker(SoxProcessor(channels=channels), get_logger("sox", filename=output_dir / "segments.log"))

    to_build: list[tuple[str, str, float, float]] = []
    for _, row in segments.iterrows():
        spk_id = str(row[GroupbyKey.SPEAKER])
        (output_dir / spk_id).mkdir(exist_ok=True)
        output = output_dir / spk_id / f"{row[GroupbyKey.SEGMENT]}.wav"
        if not output.exists():
            inp = str(input_dir / f"{row[GroupbyKey.FILE]}.wav")
            to_build.append((inp, str(output), float(row["start"]), float(row["end"])))

    launcher = Parallel(n_jobs=n_jobs, verbose=10)
    launcher(delayed(worker)(inp, output, start, end) for inp, output, start, end in to_build)
