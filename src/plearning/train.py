import json
import os
import shutil
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Optional

from plearning import CPC
from plearning.data.partition import GroupbyKey


class GPUQoS(StrEnum):
    DEFAULT = "qos_gpu-t3"
    LONGER = "qos_gpu-t4"
    DEV = "qos_gpu-dev"


JOB_HOURS = {GPUQoS.DEFAULT: 20, GPUQoS.LONGER: 100, GPUQoS.DEV: 2}


def move_train_directory(chk_source: Path, dest: Path, db_path: Path) -> None:
    """Move directory of trained model and change args accordingly"""
    dest.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(chk_source, dest / chk_source.name)
    shutil.copyfile(chk_source.parent / "checkpoint_logs.json", dest / "checkpoint_logs.json")
    with open(chk_source.parent / "checkpoint_args.json", "r") as f:
        args = json.load(f)
    with open(dest / "checkpoint_args_orig.json", "w") as f:
        json.dump(args, f, indent=2)
    args["pathDB"] = str(db_path)
    args["pathCheckpoint"] = str(dest / "checkpoint")
    args["load"] = [str(dest / chk_source.name)]
    with open(dest / "checkpoint_args.json", "w") as f:
        json.dump(args, f, indent=2)


def launch_training(
    template: Path,
    dataset: str,
    checkpoint_dir: Path,
    split: Optional[int] = None,
    idx: Optional[int] = None,
    full: bool = False,
    max_epochs: int = 100,
    groupby_key: GroupbyKey = GroupbyKey.SEGMENT,
    qos: GPUQoS = GPUQoS.DEFAULT,
    chk_to_retrain_new_dataset: Optional[Path] = None,
) -> None:
    """Launch training using a given training script on the specified dataset"""
    checkpoint_dir = checkpoint_dir.resolve()
    template = template.resolve()
    if chk_to_retrain_new_dataset is not None:
        chk_to_retrain_new_dataset = chk_to_retrain_new_dataset.resolve()

    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    assert template.is_file()
    assert (not full and (split is not None) and (idx is not None)) or (full and (split is None) and (idx is None))

    if "SCRATCH" not in os.environ:
        raise ValueError("SCRATCH not in environment variables")
    db_path = Path(dataset) / "train"
    db_path /= "full" if full else f"{groupby_key}/{split}/{idx}"
    assert (CPC.data / db_path).is_dir()

    chk_path = f"{template.stem.strip('_nodistributed')}" + f"-{str(db_path).replace('/', '_')}"
    start_time = datetime.now().strftime("%b%d_%H_%M_%S")
    identifier = f"{checkpoint_dir.name}-{chk_path}-{start_time}-%j"
    cmd = [
        "sbatch",
        "--output",
        f"./{identifier}.out",
        "--error",
        f"./{identifier}.err",
        f"--time={JOB_HOURS[qos]}:00:00",
        f"--qos={qos}",
    ]

    destination = checkpoint_dir / chk_path
    if chk_to_retrain_new_dataset is not None:
        assert chk_to_retrain_new_dataset.is_file()
        destination /= chk_to_retrain_new_dataset.parent.name
        if not destination.is_dir():
            move_train_directory(chk_to_retrain_new_dataset, destination, CPC.data / db_path)
    cmd += [str(template), str(CPC.data / db_path), str(destination), str(max_epochs)]

    if (destination / f"checkpoint_{max_epochs-1}.pt").exists():
        print(f"Already done for {destination}.")
    else:
        print(" ".join(cmd))
