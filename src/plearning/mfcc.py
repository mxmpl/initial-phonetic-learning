from pathlib import Path

from tqdm import tqdm

from plearning import CPC


def compute_mfcc(dest: Path, n_mfcc: int = 39, n_fft: int = 321) -> None:
    """Compute MFCC on all test sets using torchaudio"""
    try:
        import torch
        import torchaudio
    except ImportError as error:
        raise ImportError("You must install pytorch and torchaudio to build MFCC features.") from error

    mfcc = torchaudio.transforms.MFCC(n_mfcc=n_mfcc, melkwargs={"n_fft": n_fft})
    for test, item in CPC.test_items.items():
        assert item.is_file()
        for file in tqdm(list(item.parent.rglob(CPC.file_extension)), desc=test):
            out = (dest.resolve() / test / file.name).with_suffix(".pt")
            x, _ = torchaudio.load(file)
            y = mfcc(x).permute(0, 2, 1)
            assert len(y.shape) == 3
            assert y.shape[0] == 1
            assert y.shape[2] == n_mfcc
            torch.save(y, out)
