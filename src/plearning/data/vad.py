from pathlib import Path

from tqdm import tqdm

PYANNOTE_MODEL: str = "pyannote/voice-activity-detection"


def vad(wavs: Path, rttms: Path, hf_token: str) -> None:
    """Voice Activity Detection with pyannote.audio"""
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("Cannot run VAD: pyannote.audio is not installed")
    rttms.mkdir(exist_ok=True)
    pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=hf_token)
    for wav in tqdm(list(wavs.rglob("*.wav")), desc="VAD with pyannote.audio"):
        pyannote_vad = pipeline(wav)
        with open(rttms / f"{wav.name}.rttm", "w", encoding="utf-8") as file:
            pyannote_vad.write(file)
