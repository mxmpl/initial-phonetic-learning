# Modeling the initial state of early phonetic learning in infants

## Installation

Clone the repository and install the package with `pip`:

```bash
git clone https://github.com/mxmpl/initial-phonetic-learning
cd initial-phonetic-learning
pip install .
```

If you want to train models, evaluate them or recreate the features for the t-SNE,
you must install the [CPC3](https://github.com/bootphon/CPC3) package, with pytorch and torchaudio.

## Usage

This repository provides a CLI to build jobs to train and evaluate models with CPC3 in the context of this project, and utilities to analyze the results and reproduce the figures of the paper.

## Downloads

Training data:
- Ambient sounds: https://cognitive-ml.fr/downloads/init-plearning/train/ambient_sounds.tar (54G, md5sum: a625f3564a7697cf827d478118045c1c)
- English training set: https://cognitive-ml.fr/downloads/init-plearning/train/english.tar (53G, md5sum: 8de965e97a6dbc73fcfdc994fceed750)

Results:
- ABX scores: https://cognitive-ml.fr/downloads/init-plearning/results/abx.tar.gz (19M, md5sum: 4afecc9416484af4e7fd61483012ee6c)
- t-SNE data: https://cognitive-ml.fr/downloads/init-plearning/results/tsne.tar.gz (401M, md5sum: 13193518acb207f00858177eefc15ffb)
