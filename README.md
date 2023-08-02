# Modeling the initial state of early phonetic learning in infants

## Installation

Clone the repository and install the package with `pip`:

```bash
git clone https://github.com/mxmpl/initial-phonetic-learning
cd initial-phonetic-learning
pip install .
```

In you want to train models, evaluate them or recreate the features for the t-SNE,
you must install the [CPC3](https://github.com/bootphon/CPC3) package, with pytorch and torchaudio.

## Usage

This repository provides a CLI to build jobs to train and evaluate models with CPC3 in the context of this project, and utilities to analyse the results and reproduce the figures of the paper.

**Model checkpoints, ABX scores and t-SNE data can be found [here](https://osf.io/e4rw2)**
