import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def best_epochs(
    cpc_script: Path, root: Path, output: Path, min_epoch: Optional[int] = None, max_epoch: Optional[int] = None
) -> None:
    """Find best epoch of each experiment based on the accuracy on the validation set"""
    model, epoch = [], []
    cmd = ["python", str(cpc_script.resolve())]
    if min_epoch is not None:
        cmd += ["--min", str(min_epoch)]
    if max_epoch is not None:
        cmd += ["--max", str(max_epoch)]
    cmd.append("--model_path")
    checkpoints = np.unique([path.parent for path in root.resolve().rglob("*.pt")])
    for directory in tqdm(checkpoints):
        res = subprocess.run(cmd + [directory], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            print(f"Failed for {directory}")
            continue
        model.append(str(directory))
        epoch.append(int(res.stdout.split(" ")[-1]))
    pd.DataFrame({"model": model, "epoch": epoch}).to_csv(output, index=False)
