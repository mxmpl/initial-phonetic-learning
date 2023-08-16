import dataclasses
import sys
from pathlib import Path
from typing import Callable, Generator, Union

import pandas as pd

from plearning import CPC


def get_last_parts(path: Path, depth: int) -> Path:
    return Path("/".join([part for part in path.parts[-depth - 1 :]]))


def get_last_dirs(path: Path) -> list[Path]:
    assert sys.version_info >= (3, 11), "Need python 3.11 for the following pathlib method"
    subdirs = list(path.glob("*/"))
    if len(subdirs) == 0:
        return [path]
    return sum([get_last_dirs(subdir) for subdir in subdirs], [])


@dataclasses.dataclass
class Evaluator:
    cmd_func: Callable[..., str]
    generator: Callable[[Path], Generator[tuple[Path, Path], None, None]]

    def _make_jobs(self, root: Path, item: Path) -> list[tuple[str, bool]]:
        assert item.is_file()
        cmds_to_run = []
        for checkpoint, out_checkpoint in self.generator(root):
            out_checkpoint.mkdir(parents=True, exist_ok=True)
            for mode in ["within", "across"]:
                cmd = self.cmd_func(checkpoint=checkpoint, item=item, mode=mode, out=out_checkpoint / mode)
                if all([(out_checkpoint / mode / file).is_file() for file in CPC.evaluation_files]):
                    cmds_to_run.append((cmd, False))
                else:
                    cmds_to_run.append((cmd, True))
        return cmds_to_run

    def __call__(self, output: Path) -> None:
        output = output.resolve()
        cmds: list[tuple[str, bool]] = sum(
            [self._make_jobs(output / test, item) for (test, item) in CPC.test_items.items()],
            [],
        )
        (output / "cmd.sh").write_text("\n".join([cmd for cmd, _ in cmds]) + "\n")
        (output / "to_run.sh").write_text("\n".join([cmd for cmd, to_run in cmds if to_run]) + "\n")


def evaluate_cpc_best_epochs(
    eval_abx: Path, output: Path, checkpoints: Path, best_epochs: Path, max_epochs: int = 100, hierarchy_depth: int = 1
) -> None:
    """Create jobs to evaluate all models given their best epoch on the validation set"""
    eval_abx = eval_abx.resolve()
    checkpoints = checkpoints.resolve()
    assert eval_abx.is_file()
    assert checkpoints.is_dir()
    assert hierarchy_depth >= 1

    def generator(out: Path) -> Generator[tuple[Path, Path], None, None]:
        epochs = pd.read_csv(best_epochs.resolve()).set_index("model")
        for chk in checkpoints.rglob(f"checkpoint_{max_epochs - 1}.pt"):
            checkpoint = chk.parent / "checkpoint_{}.pt".format(int(epochs.loc[str(chk.parent)]))
            out_checkpoint = out / get_last_parts(checkpoint, hierarchy_depth).with_suffix("")
            yield checkpoint, out_checkpoint

    def cmd_func(**kwargs: Union[Path, str]) -> str:
        return CPC.evaluation().format(
            eval_abx=eval_abx,
            dataset=Path(kwargs["item"]).parent,
            **kwargs,
        )

    Evaluator(cmd_func, generator)(output)


def evaluate_cpc_pre_computed(eval_abx: Path, output: Path, features: Path) -> None:
    """Create jobs to evaluate pre-computed CPC features"""
    eval_abx = eval_abx.resolve()
    features = features.resolve()
    assert eval_abx.is_file()
    assert features.is_dir()

    def generator(out: Path) -> Generator[tuple[Path, Path], None, None]:
        for feats in get_last_dirs(features):
            last_part = Path("/".join(feats.parts[len(features.parts) :]))
            yield feats, out / last_part

    def cmd_func(**kwargs: Union[Path, str]) -> str:
        return CPC.evaluation_from_pre_computed().format(eval_abx=eval_abx, dataset=kwargs["checkpoint"], **kwargs)

    Evaluator(cmd_func, generator)(output)


def evaluate_mfcc(eval_abx: Path, output: Path, mfcc: Path) -> None:
    """Create jobs to evaluate pre-computed MFCC"""
    eval_abx = eval_abx.resolve()
    mfcc = mfcc.resolve()
    assert eval_abx.is_file()
    assert mfcc.is_dir()

    def generator(out: Path) -> Generator[tuple[Path, Path], None, None]:
        yield mfcc / out.stem, out

    def cmd_func(**kwargs: Union[Path, str]) -> str:
        return CPC.evaluation_from_pre_computed().format(eval_abx=eval_abx, dataset=kwargs["checkpoint"], **kwargs)

    Evaluator(cmd_func, generator)(output)
