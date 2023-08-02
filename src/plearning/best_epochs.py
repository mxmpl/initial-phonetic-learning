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
    results = {"model": [], "epoch": []}
    cmd = ["python", str(cpc_script.resolve()), "--model_path", None]
    if min_epoch is not None:
        cmd += ["--min", str(min_epoch)]
    if max_epoch is not None:
        cmd += ["--max", str(max_epoch)]
    checkpoints = np.unique([path.parent for path in root.resolve().rglob("*.pt")])
    for directory in tqdm(checkpoints):
        cmd[3] = directory
        res = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            print(f"Failed for {directory}")
            continue
        results["model"].append(str(directory))
        results["epoch"].append(int(res.stdout.split(" ")[-1]))
    pd.DataFrame(results).to_csv(output, index=False)
