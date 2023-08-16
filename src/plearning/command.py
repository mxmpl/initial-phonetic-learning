import dataclasses
import os
from pathlib import Path
from typing import Optional


@dataclasses.dataclass(frozen=True)
class CPCCommands:
    seq_norm: bool = True
    strict: bool = True
    max_size_seq: int = 64000
    file_extension: str = ".wav"
    data: Optional[Path] = None

    evaluation_files: list[str] = dataclasses.field(
        default_factory=lambda: ["ABX_args.json", "ABX_scores.json", "extras.pkl"]
    )

    @property
    def test_items(self) -> dict[str, Path]:
        if self.data is None:
            try:
                scratch = os.environ["SCRATCH"]
            except KeyError as error:
                raise KeyError(
                    "Must set SCRATCH environment variable if you use CPCCommands without specifying `data`."
                ) from error
            data_root = Path(scratch).resolve()
        else:
            data_root = Path(self.data).resolve()
        test_items = {
            "csj": data_root / "data/CSJ/test/test.item",
            "gpj": data_root / "data/GPJ/test/test.item",
            "buc": data_root / "data/BUC/test/test.item",
            "wsj": data_root / "data/WSJ/test/test.item",
        }
        for test, path in test_items.items():
            assert path.is_file(), test
        return test_items

    @property
    def appendix(self) -> str:
        cmd = ""
        if self.seq_norm:
            cmd += "--seq_norm "
        if self.strict:
            cmd += "--strict "
        cmd += f"--max_size_seq {self.max_size_seq}"
        return cmd

    def evaluation(self) -> str:
        cmd = "python {eval_abx} from_checkpoint {checkpoint} {item} {dataset} --mode {mode} --out {out} --cuda "
        return cmd + self.appendix + f" --file_extension {self.file_extension} "

    def evaluation_from_pre_computed(self) -> str:
        return "python {eval_abx} from_pre_computed {item} {dataset} --file_extension .pt --mode {mode} --out {out}"
