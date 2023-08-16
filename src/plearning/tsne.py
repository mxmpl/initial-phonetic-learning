from collections import namedtuple
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from plearning import CPC


def build_tsne(
    path_checkpoint: str,
    path_dataset: str,
    path_item_file: str,
    output: Path,
    feature_size: float = 0.01,
    get_encoded: bool = False,
    from_centroid: Optional[str] = None,
    perplexity: float = 30.0,
    seed: int = 0,
    is_untrained: bool = False,
) -> None:
    """Compute output features and their t-SNE representations of a given model and test set"""
    try:
        import torch
    except ImportError as error:
        raise ImportError("You must install pytorch to build features for t-SNE") from error

    try:
        from cpc.clustering.clustering import loadClusterModule
        from cpc.dataset import findAllSeqs
        from cpc.eval.ABX import abx_iterators as abx_it
        from cpc.feature_loader import FeatureModule, buildFeature, loadModel
    except ImportError as error:
        raise ImportError("You must install CPC3 to build features for t-SNE") from error

    model = loadModel([path_checkpoint], loadStateDict=not is_untrained)[0]
    model.gAR.keepHidden = True
    feature_maker = FeatureModule(model, get_encoded).cuda().eval()

    def base_features(x: str) -> torch.Tensor:
        return buildFeature(feature_maker, x, seqNorm=CPC.seq_norm, strict=CPC.strict, maxSizeSeq=CPC.max_size_seq)

    if from_centroid is not None:
        cluster_module = loadClusterModule(from_centroid)

        def feature_function(x: str) -> torch.Tensor:
            c_feature = base_features(x)
            dist_clusters = cluster_module(c_feature)
            q_feature = torch.argmin(dist_clusters, dim=-1)
            return cluster_module.Ck[:, q_feature.squeeze()]

    else:
        feature_function = base_features

    seq_list, _ = findAllSeqs(path_dataset, extension=CPC.file_extension)
    seq_list = [(str(Path(x).stem), str(Path(path_dataset) / x)) for (_, x) in seq_list]

    output.mkdir(exist_ok=True)
    dataset = abx_it.ABXFeatureLoader(path_item_file, seq_list, feature_function, 1 / feature_size, normalize=True)
    context_match = {v: k for k, v in dataset.context_match.items()}
    phone_match = {v: k for k, v in dataset.phone_match.items()}
    speaker_match = {v: k for k, v in dataset.speaker_match.items()}
    full_data, df = [], []

    PhoneInfos = namedtuple("PhoneInfos", ["idx", "context", "phone", "speaker"])
    for idx, (sample_data, _, (context_id, phone_id, speaker_id)) in enumerate(dataset):
        full_data.append(sample_data.mean(axis=0))
        df.append(PhoneInfos(idx, context_match[context_id], phone_match[phone_id], speaker_match[speaker_id]))
    data = torch.stack(full_data).numpy()
    phone_infos = pd.DataFrame(df)

    np.save(output / "data.npy", data)
    phone_infos.to_csv(output / "phone_infos.csv", index=False)

    data_embedded = TSNE(perplexity=perplexity, verbose=10, random_state=seed).fit_transform(data)
    np.save(output / "tsne.npy", data_embedded)

    for phone_pair in ["rl", "wy"]:
        data_pair = data[data.phone.isin([*phone_pair.upper()])]
        data_embedded = TSNE(perplexity=perplexity, verbose=10, random_state=seed).fit_transform(data_pair)
        np.save(output / f"tsne_{phone_pair}.npy", data_embedded)
