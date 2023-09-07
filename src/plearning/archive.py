import json
import shutil
from pathlib import Path

import pandas as pd


def copy_checkpoint(src: Path, dest: Path) -> None:
    dest.symlink_to(src)
    shutil.copyfile(src.parent / "checkpoint_logs.json", dest.parent / "checkpoint_logs.json")
    with open(src.parent / "checkpoint_args.json", "r") as file:
        args = json.load(file)
    for key in [
        "pathDB",
        "pathCheckpoint",
        "nGPU",
        "load",
        "master_addr",
        "master_port",
        "distributed",
        "world_size",
        "global_rank",
        "local_rank",
        "n_nodes",
        "node_id",
        "n_gpu_per_node",
    ]:
        args[key] = None
    with open(dest.parent / "checkpoint_args.json", "w") as file:
        json.dump(args, file, indent=2)


def archive(checkpoints: Path, destination: Path, best_epochs: Path) -> None:
    """Copy the best checkpoints of the models listed in the `best_epochs` file"""
    df = pd.read_csv(best_epochs)
    checkpoints, destination = checkpoints.resolve(), destination.resolve()
    paths = [Path(path) for path in df.apply(lambda row: Path(row.model) / f"checkpoint_{row.epoch}.pt", axis=1)]
    for src in paths:
        assert src.is_relative_to(checkpoints), (src, checkpoints)
        dest = destination / src.relative_to(checkpoints)
        dest.parent.mkdir(exist_ok=True, parents=True)
        copy_checkpoint(src, dest)
