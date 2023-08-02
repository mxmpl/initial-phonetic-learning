# Notebooks and results

Those notebooks are to reproduce the Figures 2-6 of the paper.
Download the CSV files containing the results before using them.

The `paper.mplstyle` is the Matplotlib style file.
Use it only if you have LaTeX installed with the correct packages.

## Download ABX scores and t-SNE representations

- Go to https://osf.io/e4rw2, and download `abx.tar.gz` and `tsne.tar.gz` in this folder.
- Run `tar xfz abx.tar.gz` and `tar xfz tsne.tar.gz`
- Now you can reproduce the figures using the notebooks `trajectories.ipynb` and `tsne.ipynb`

## Files specification

`abx`:
  - `initial_state_untrained.csv`: ABX error rates for the initial state without pretraining (ie. untrained model).
  - `initial_state_noise.csv`: same but for model trained on 500h of ambient sounds.
  - `no_pretraining.csv`: errors for all the models without pretraining
  - `noise_pretraining.csv`: same but for models pretrained on 500h of ambient sounds.
  - `crossling_pretraining.csv`: same but for models with cross-lingual pretraining on 500h of speech.
  - `mfcc.csv`: ABX error rates of MFCC.

`tsne`:
  - `untrained` or `noise`: either from the initial state without pretraining or the model trained on 500h of ambient sounds.
    - `tsne.npy`: t-SNE fitted on all phones representations of the Buckeye corpus.
    - `tsne_rl.npy`: same but on all [ɹ] and [l] only.
    - `tsne_wj.npy`: same but on all [w] and [j] only.

## Table of scores

Full tables of scores for the trained models. For each configuration, the mean (std) accuracy
across the models trained is reported (either 1, 5 or 15 models depending on the training duration).

Run `prettier_scores.py` to regenerate those tables.

### Within speakers

#### All contrasts

| Pretraining    | Training      | Japanese (CSJ / GPJ)    | English (Buckeye / WSJ) |
| :------------- | :------------ | :---------------------- | :---------------------- |
| No pretraining | English 500h  | 95.2 / 95.5             | 93.0 / 96.4             |
| No pretraining | English 100h  | 94.5 (0.0) / 95.1 (0.1) | 92.4 (0.1) / 95.7 (0.0) |
| No pretraining | English 20h   | 94.1 (0.1) / 94.8 (0.1) | 91.8 (0.1) / 95.1 (0.1) |
| No pretraining | English 4h    | 92.2 (0.3) / 93.5 (0.2) | 89.2 (0.2) / 92.9 (0.2) |
| No pretraining | English 1h    | 88.7 (0.5) / 90.9 (0.3) | 84.3 (0.4) / 87.8 (0.7) |
| No pretraining | Japanese 500h | 96.1 / 95.5             | 92.0 / 94.7             |
| No pretraining | Japanese 100h | 95.9 (0.1) / 95.9 (0.1) | 92.2 (0.1) / 94.9 (0.1) |
| No pretraining | Japanese 20h  | 95.0 (0.6) / 95.2 (0.4) | 91.4 (0.5) / 94.2 (0.3) |
| No pretraining | Japanese 4h   | 92.6 (1.3) / 93.6 (0.9) | 89.1 (1.3) / 92.3 (1.4) |
| No pretraining | Japanese 1h   | 89.4 (0.4) / 91.3 (0.4) | 85.5 (0.5) / 89.5 (0.6) |
| Cross-lingual  | English 500h  | 94.5 / 94.9             | 92.7 / 96.3             |
| Cross-lingual  | English 100h  | 94.7 (0.1) / 95.2 (0.1) | 92.7 (0.1) / 96.1 (0.0) |
| Cross-lingual  | English 20h   | 94.7 (0.1) / 95.1 (0.1) | 92.3 (0.1) / 95.7 (0.1) |
| Cross-lingual  | English 4h    | 95.0 (0.1) / 95.3 (0.1) | 91.9 (0.1) / 94.9 (0.2) |
| Cross-lingual  | English 1h    | 95.3 (0.1) / 95.3 (0.1) | 91.7 (0.1) / 94.5 (0.2) |
| Cross-lingual  | Japanese 500h | 96.3 / 96.0             | 92.3 / 95.2             |
| Cross-lingual  | Japanese 100h | 96.3 (0.0) / 95.9 (0.1) | 92.3 (0.0) / 95.3 (0.1) |
| Cross-lingual  | Japanese 20h  | 96.0 (0.1) / 95.9 (0.1) | 92.2 (0.1) / 95.2 (0.2) |
| Cross-lingual  | Japanese 4h   | 95.6 (0.0) / 95.8 (0.1) | 92.4 (0.1) / 95.6 (0.2) |
| Cross-lingual  | Japanese 1h   | 95.3 (0.1) / 95.5 (0.0) | 92.5 (0.1) / 95.7 (0.1) |
| Ambient sounds | English 500h  | 94.3 / 94.9             | 92.1 / 95.5             |
| Ambient sounds | English 100h  | 93.9 (0.1) / 94.5 (0.1) | 91.5 (0.1) / 95.0 (0.1) |
| Ambient sounds | English 20h   | 93.9 (0.1) / 94.4 (0.1) | 91.3 (0.1) / 94.9 (0.1) |
| Ambient sounds | English 4h    | 92.9 (0.1) / 93.7 (0.1) | 90.1 (0.1) / 94.1 (0.1) |
| Ambient sounds | English 1h    | 92.5 (0.2) / 93.3 (0.2) | 89.5 (0.3) / 93.8 (0.2) |
| Ambient sounds | Japanese 500h | 95.6 / 95.4             | 91.8 / 94.7             |
| Ambient sounds | Japanese 100h | 95.2 (0.1) / 95.3 (0.0) | 91.6 (0.1) / 94.7 (0.1) |
| Ambient sounds | Japanese 20h  | 94.6 (0.2) / 94.9 (0.1) | 91.1 (0.2) / 94.2 (0.2) |
| Ambient sounds | Japanese 4h   | 93.3 (0.2) / 94.1 (0.2) | 90.1 (0.2) / 93.7 (0.3) |
| Ambient sounds | Japanese 1h   | 92.5 (0.7) / 93.5 (0.6) | 89.4 (1.0) / 93.3 (1.3) |

#### [ɹ]-[l]

| Pretraining    | Training      | English (Buckeye / WSJ) |
| :------------- | :------------ | :---------------------- |
| No pretraining | English 500h  | 92.8 / 97.3             |
| No pretraining | English 100h  | 91.9 (0.7) / 95.7 (0.4) |
| No pretraining | English 20h   | 90.5 (0.4) / 94.7 (0.4) |
| No pretraining | English 4h    | 82.0 (1.2) / 89.1 (1.2) |
| No pretraining | English 1h    | 65.4 (1.3) / 72.9 (1.0) |
| No pretraining | Japanese 500h | 87.3 / 92.0             |
| No pretraining | Japanese 100h | 88.5 (0.3) / 93.5 (0.5) |
| No pretraining | Japanese 20h  | 87.2 (0.9) / 93.1 (0.6) |
| No pretraining | Japanese 4h   | 82.1 (3.7) / 89.0 (3.1) |
| No pretraining | Japanese 1h   | 71.2 (1.2) / 80.2 (1.8) |
| Cross-lingual  | English 500h  | 92.0 / 96.4             |
| Cross-lingual  | English 100h  | 91.6 (0.3) / 96.0 (0.3) |
| Cross-lingual  | English 20h   | 90.3 (0.4) / 94.8 (0.3) |
| Cross-lingual  | English 4h    | 87.4 (0.4) / 92.9 (0.4) |
| Cross-lingual  | English 1h    | 86.2 (0.4) / 92.1 (0.3) |
| Cross-lingual  | Japanese 500h | 88.6 / 93.1             |
| Cross-lingual  | Japanese 100h | 88.4 (0.1) / 93.9 (0.3) |
| Cross-lingual  | Japanese 20h  | 88.9 (0.5) / 93.9 (0.5) |
| Cross-lingual  | Japanese 4h   | 90.3 (0.4) / 94.9 (0.3) |
| Cross-lingual  | Japanese 1h   | 90.8 (0.5) / 95.3 (0.4) |
| Ambient sounds | English 500h  | 91.5 / 96.2             |
| Ambient sounds | English 100h  | 89.6 (0.2) / 94.8 (0.3) |
| Ambient sounds | English 20h   | 90.3 (0.4) / 95.3 (0.3) |
| Ambient sounds | English 4h    | 88.6 (0.5) / 94.2 (0.3) |
| Ambient sounds | English 1h    | 88.3 (0.5) / 93.9 (0.5) |
| Ambient sounds | Japanese 500h | 88.1 / 93.5             |
| Ambient sounds | Japanese 100h | 88.3 (0.1) / 93.8 (0.3) |
| Ambient sounds | Japanese 20h  | 88.6 (0.4) / 94.1 (0.3) |
| Ambient sounds | Japanese 4h   | 88.4 (0.6) / 93.4 (0.6) |
| Ambient sounds | Japanese 1h   | 88.0 (2.0) / 92.9 (2.5) |

#### [w]-[j]

| Pretraining    | Training      | Japanese (CSJ / GPJ)    | English (Buckeye / WSJ) |
| :------------- | :------------ | :---------------------- | :---------------------- |
| No pretraining | English 500h  | 90.3 / 94.0             | 94.2 / 98.5             |
| No pretraining | English 100h  | 90.7 (0.6) / 94.0 (0.1) | 94.2 (0.2) / 98.2 (0.3) |
| No pretraining | English 20h   | 90.5 (0.5) / 94.0 (0.4) | 93.8 (0.5) / 97.7 (0.4) |
| No pretraining | English 4h    | 87.3 (1.0) / 92.5 (0.5) | 91.3 (0.9) / 94.5 (1.4) |
| No pretraining | English 1h    | 74.3 (3.3) / 83.7 (2.3) | 75.7 (3.3) / 82.8 (2.9) |
| No pretraining | Japanese 500h | 91.1 / 94.5             | 93.7 / 95.9             |
| No pretraining | Japanese 100h | 91.6 (0.1) / 95.1 (0.2) | 94.1 (0.3) / 97.8 (0.4) |
| No pretraining | Japanese 20h  | 91.0 (1.1) / 94.5 (0.6) | 93.6 (0.5) / 97.4 (0.6) |
| No pretraining | Japanese 4h   | 87.7 (2.5) / 92.7 (1.6) | 90.9 (2.6) / 95.2 (3.0) |
| No pretraining | Japanese 1h   | 80.3 (1.6) / 88.9 (1.0) | 83.3 (1.8) / 90.2 (1.8) |
| Cross-lingual  | English 500h  | 89.7 / 94.0             | 93.3 / 97.9             |
| Cross-lingual  | English 100h  | 89.6 (0.3) / 94.0 (0.1) | 93.7 (0.3) / 98.2 (0.3) |
| Cross-lingual  | English 20h   | 90.1 (0.4) / 94.0 (0.3) | 93.3 (0.2) / 98.0 (0.4) |
| Cross-lingual  | English 4h    | 90.6 (0.3) / 94.5 (0.2) | 93.2 (0.3) / 97.1 (0.6) |
| Cross-lingual  | English 1h    | 90.7 (0.3) / 94.4 (0.3) | 93.6 (0.4) / 96.6 (0.7) |
| Cross-lingual  | Japanese 500h | 91.8 / 95.6             | 93.7 / 97.4             |
| Cross-lingual  | Japanese 100h | 91.8 (0.1) / 95.4 (0.1) | 93.8 (0.1) / 97.4 (0.3) |
| Cross-lingual  | Japanese 20h  | 91.9 (0.4) / 95.2 (0.2) | 93.5 (0.3) / 97.4 (0.6) |
| Cross-lingual  | Japanese 4h   | 91.2 (0.3) / 95.0 (0.3) | 93.9 (0.3) / 97.5 (0.3) |
| Cross-lingual  | Japanese 1h   | 91.4 (0.3) / 95.0 (0.3) | 94.1 (0.5) / 97.9 (0.4) |
| Ambient sounds | English 500h  | 89.4 / 93.5             | 92.7 / 98.2             |
| Ambient sounds | English 100h  | 89.0 (0.3) / 93.6 (0.2) | 92.3 (0.2) / 97.6 (0.3) |
| Ambient sounds | English 20h   | 89.4 (0.5) / 93.6 (0.3) | 91.8 (0.3) / 97.2 (0.3) |
| Ambient sounds | English 4h    | 87.6 (0.5) / 92.2 (0.4) | 89.9 (0.5) / 96.5 (0.4) |
| Ambient sounds | English 1h    | 87.3 (0.6) / 91.6 (0.4) | 89.3 (0.8) / 96.5 (0.7) |
| Ambient sounds | Japanese 500h | 90.0 / 94.3             | 93.4 / 97.3             |
| Ambient sounds | Japanese 100h | 90.1 (0.5) / 94.6 (0.2) | 93.3 (0.3) / 97.6 (0.4) |
| Ambient sounds | Japanese 20h  | 90.6 (0.4) / 94.5 (0.3) | 92.5 (0.4) / 97.4 (0.4) |
| Ambient sounds | Japanese 4h   | 89.6 (0.3) / 93.5 (0.3) | 90.9 (0.7) / 96.5 (1.0) |
| Ambient sounds | Japanese 1h   | 88.9 (0.8) / 92.9 (0.6) | 89.8 (2.3) / 96.1 (2.6) |

### Across speakers

#### All contrasts

| Pretraining    | Training      | Japanese (CSJ / GPJ)    | English (Buckeye / WSJ) |
| :------------- | :------------ | :---------------------- | :---------------------- |
| No pretraining | English 500h  | 92.7 / 93.3             | 91.0 / 95.5             |
| No pretraining | English 100h  | 91.7 (0.1) / 92.8 (0.1) | 90.0 (0.1) / 94.3 (0.1) |
| No pretraining | English 20h   | 91.0 (0.1) / 92.2 (0.1) | 89.1 (0.2) / 93.5 (0.2) |
| No pretraining | English 4h    | 88.4 (0.3) / 89.9 (0.2) | 85.6 (0.3) / 90.5 (0.3) |
| No pretraining | English 1h    | 84.3 (0.5) / 85.9 (0.4) | 79.8 (0.5) / 84.3 (0.8) |
| No pretraining | Japanese 500h | 94.5 / 93.7             | 89.4 / 93.0             |
| No pretraining | Japanese 100h | 94.1 (0.1) / 94.4 (0.1) | 89.5 (0.1) / 93.1 (0.1) |
| No pretraining | Japanese 20h  | 92.8 (0.8) / 93.3 (0.6) | 88.5 (0.6) / 92.1 (0.5) |
| No pretraining | Japanese 4h   | 89.3 (1.6) / 90.5 (1.3) | 85.6 (1.5) / 89.4 (1.8) |
| No pretraining | Japanese 1h   | 85.2 (0.5) / 87.1 (0.5) | 81.3 (0.6) / 85.8 (0.9) |
| Cross-lingual  | English 500h  | 92.0 / 92.8             | 90.8 / 95.5             |
| Cross-lingual  | English 100h  | 92.3 (0.1) / 93.0 (0.0) | 90.6 (0.1) / 95.3 (0.1) |
| Cross-lingual  | English 20h   | 92.3 (0.1) / 92.9 (0.1) | 90.0 (0.1) / 94.5 (0.1) |
| Cross-lingual  | English 4h    | 92.7 (0.2) / 93.3 (0.2) | 89.4 (0.2) / 93.4 (0.3) |
| Cross-lingual  | English 1h    | 93.1 (0.2) / 93.5 (0.1) | 89.1 (0.2) / 92.7 (0.3) |
| Cross-lingual  | Japanese 500h | 94.9 / 94.4             | 89.5 / 93.4             |
| Cross-lingual  | Japanese 100h | 94.7 (0.0) / 94.1 (0.1) | 89.4 (0.1) / 93.7 (0.1) |
| Cross-lingual  | Japanese 20h  | 94.2 (0.1) / 94.1 (0.1) | 89.2 (0.2) / 93.5 (0.3) |
| Cross-lingual  | Japanese 4h   | 93.4 (0.1) / 93.9 (0.1) | 89.7 (0.2) / 94.0 (0.2) |
| Cross-lingual  | Japanese 1h   | 92.9 (0.1) / 93.5 (0.1) | 90.0 (0.1) / 94.4 (0.1) |
| Ambient sounds | English 500h  | 91.0 / 92.0             | 89.2 / 93.9             |
| Ambient sounds | English 100h  | 89.9 (0.2) / 91.3 (0.1) | 87.9 (0.1) / 92.9 (0.1) |
| Ambient sounds | English 20h   | 89.7 (0.1) / 91.0 (0.1) | 87.5 (0.1) / 92.5 (0.2) |
| Ambient sounds | English 4h    | 87.7 (0.2) / 89.4 (0.2) | 85.2 (0.3) / 90.8 (0.2) |
| Ambient sounds | English 1h    | 86.8 (0.2) / 88.4 (0.2) | 84.1 (0.2) / 90.2 (0.2) |
| Ambient sounds | Japanese 500h | 93.3 / 93.3             | 88.7 / 92.2             |
| Ambient sounds | Japanese 100h | 92.5 (0.1) / 92.8 (0.1) | 88.2 (0.1) / 92.3 (0.1) |
| Ambient sounds | Japanese 20h  | 91.2 (0.2) / 91.8 (0.2) | 86.9 (0.3) / 91.4 (0.3) |
| Ambient sounds | Japanese 4h   | 88.5 (0.2) / 89.9 (0.3) | 85.2 (0.2) / 90.4 (0.4) |
| Ambient sounds | Japanese 1h   | 87.2 (0.6) / 88.7 (0.6) | 84.1 (0.8) / 89.4 (1.4) |

#### [ɹ]-[l]

| Pretraining    | Training      | English (Buckeye / WSJ) |
| :------------- | :------------ | :---------------------- |
| No pretraining | English 500h  | 89.3 / 95.1             |
| No pretraining | English 100h  | 88.3 (1.1) / 93.0 (0.7) |
| No pretraining | English 20h   | 86.7 (0.5) / 91.5 (0.7) |
| No pretraining | English 4h    | 75.9 (1.5) / 82.8 (1.6) |
| No pretraining | English 1h    | 58.3 (1.2) / 64.2 (1.1) |
| No pretraining | Japanese 500h | 80.0 / 87.8             |
| No pretraining | Japanese 100h | 82.5 (0.6) / 89.8 (0.7) |
| No pretraining | Japanese 20h  | 81.5 (1.1) / 88.9 (0.9) |
| No pretraining | Japanese 4h   | 75.7 (4.1) / 83.2 (4.1) |
| No pretraining | Japanese 1h   | 64.0 (1.2) / 72.1 (2.2) |
| Cross-lingual  | English 500h  | 88.2 / 94.4             |
| Cross-lingual  | English 100h  | 87.7 (0.3) / 93.6 (0.4) |
| Cross-lingual  | English 20h   | 86.0 (0.5) / 91.7 (0.4) |
| Cross-lingual  | English 4h    | 81.5 (0.7) / 88.4 (0.6) |
| Cross-lingual  | English 1h    | 79.4 (0.5) / 86.8 (0.5) |
| Cross-lingual  | Japanese 500h | 80.8 / 88.3             |
| Cross-lingual  | Japanese 100h | 80.8 (0.5) / 88.7 (0.4) |
| Cross-lingual  | Japanese 20h  | 82.1 (0.9) / 89.3 (1.0) |
| Cross-lingual  | Japanese 4h   | 84.5 (0.6) / 91.1 (0.5) |
| Cross-lingual  | Japanese 1h   | 85.6 (0.7) / 92.0 (0.4) |
| Ambient sounds | English 500h  | 86.1 / 92.3             |
| Ambient sounds | English 100h  | 82.6 (0.7) / 89.1 (0.5) |
| Ambient sounds | English 20h   | 83.5 (0.6) / 90.1 (0.6) |
| Ambient sounds | English 4h    | 79.2 (1.0) / 86.3 (0.7) |
| Ambient sounds | English 1h    | 78.5 (0.8) / 85.5 (0.7) |
| Ambient sounds | Japanese 500h | 80.6 / 88.0             |
| Ambient sounds | Japanese 100h | 80.4 (0.2) / 88.1 (0.4) |
| Ambient sounds | Japanese 20h  | 80.6 (0.7) / 87.9 (0.7) |
| Ambient sounds | Japanese 4h   | 79.4 (0.8) / 86.0 (0.8) |
| Ambient sounds | Japanese 1h   | 77.9 (2.0) / 84.6 (2.6) |

#### [w]-[j]

| Pretraining    | Training      | Japanese (CSJ / GPJ)    | English (Buckeye / WSJ) |
| :------------- | :------------ | :---------------------- | :---------------------- |
| No pretraining | English 500h  | 87.8 / 91.4             | 91.3 / 97.2             |
| No pretraining | English 100h  | 88.1 (0.8) / 92.2 (0.3) | 91.7 (0.4) / 96.4 (0.3) |
| No pretraining | English 20h   | 88.0 (0.8) / 92.2 (0.6) | 91.0 (0.7) / 95.8 (0.5) |
| No pretraining | English 4h    | 84.2 (1.0) / 90.8 (0.6) | 87.8 (1.1) / 92.5 (1.5) |
| No pretraining | English 1h    | 70.9 (3.8) / 80.8 (2.6) | 70.4 (3.7) / 78.8 (3.1) |
| No pretraining | Japanese 500h | 89.2 / 92.9             | 91.3 / 95.6             |
| No pretraining | Japanese 100h | 89.8 (0.3) / 93.5 (0.4) | 91.8 (0.5) / 96.8 (0.3) |
| No pretraining | Japanese 20h  | 88.7 (1.1) / 93.2 (0.5) | 91.3 (0.7) / 96.3 (0.5) |
| No pretraining | Japanese 4h   | 84.8 (3.0) / 91.2 (2.0) | 88.1 (2.9) / 92.8 (3.9) |
| No pretraining | Japanese 1h   | 76.7 (1.8) / 86.9 (1.2) | 79.4 (2.0) / 86.2 (2.6) |
| Cross-lingual  | English 500h  | 87.0 / 91.7             | 90.3 / 97.1             |
| Cross-lingual  | English 100h  | 87.0 (0.3) / 91.9 (0.2) | 90.5 (0.4) / 96.8 (0.2) |
| Cross-lingual  | English 20h   | 87.4 (0.5) / 92.0 (0.5) | 90.5 (0.3) / 96.5 (0.4) |
| Cross-lingual  | English 4h    | 88.5 (0.5) / 92.9 (0.3) | 90.4 (0.3) / 96.2 (0.3) |
| Cross-lingual  | English 1h    | 88.8 (0.6) / 93.0 (0.4) | 90.7 (0.5) / 95.9 (0.6) |
| Cross-lingual  | Japanese 500h | 90.9 / 94.2             | 91.3 / 96.6             |
| Cross-lingual  | Japanese 100h | 90.1 (0.2) / 93.9 (0.1) | 91.1 (0.3) / 96.5 (0.3) |
| Cross-lingual  | Japanese 20h  | 89.5 (0.4) / 93.5 (0.3) | 90.5 (0.5) / 96.4 (0.5) |
| Cross-lingual  | Japanese 4h   | 88.6 (0.5) / 93.0 (0.4) | 90.9 (0.6) / 96.4 (0.3) |
| Cross-lingual  | Japanese 1h   | 89.1 (0.5) / 92.9 (0.5) | 91.5 (0.7) / 96.7 (0.3) |
| Ambient sounds | English 500h  | 85.4 / 89.8             | 88.7 / 96.2             |
| Ambient sounds | English 100h  | 84.1 (0.4) / 89.5 (0.5) | 87.8 (0.4) / 95.2 (0.6) |
| Ambient sounds | English 20h   | 84.4 (0.7) / 89.2 (0.6) | 86.9 (0.6) / 94.3 (0.5) |
| Ambient sounds | English 4h    | 80.9 (0.9) / 86.5 (0.6) | 83.5 (0.8) / 92.3 (0.7) |
| Ambient sounds | English 1h    | 79.8 (1.1) / 85.3 (0.9) | 82.5 (1.0) / 91.4 (0.9) |
| Ambient sounds | Japanese 500h | 87.0 / 91.8             | 90.6 / 95.0             |
| Ambient sounds | Japanese 100h | 86.8 (0.8) / 91.7 (0.4) | 89.5 (0.4) / 95.9 (0.4) |
| Ambient sounds | Japanese 20h  | 86.8 (0.6) / 91.7 (0.4) | 88.4 (0.7) / 94.7 (0.6) |
| Ambient sounds | Japanese 4h   | 83.7 (0.7) / 89.0 (0.6) | 85.7 (1.0) / 92.7 (1.3) |
| Ambient sounds | Japanese 1h   | 82.9 (1.2) / 88.0 (0.9) | 83.9 (2.5) / 91.2 (2.5) |
