# English training dataset

The full dataset is available [here](https://cognitive-ml.fr/downloads/init-plearning/train/english.tar) (53G, md5sum: 8de965e97a6dbc73fcfdc994fceed750).

If you want to reproduce the experiments either:
- Move the `full` folder extracted from `english.tar` to `$DATASET/train/full`.
- Or rebuild this dataset from LibriVox:
  - Use the recipes of [Libri-Light](https://github.com/facebookresearch/libri-light) to download and segment data.
  - Then select the audio files using using `matching_selection.py`

Then use the following command to split the full dataset in smaller subsets:
```bash
plearning data partition $DATASET/train/full $DATASET/csv/train_segments.csv
```
