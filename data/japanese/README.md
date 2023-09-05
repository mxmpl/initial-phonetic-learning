# Corpus of Spontaneous Japanese

The Japanese training set is derived from the [Corpus of Spontaneous Japanese](https://clrd.ninjal.ac.jp/csj/en).
If you have access to the corpus, you can follow these steps to recreate our datasets.

## Recreate the datasets

First, the test set is built following the recipes of [1] using their companion repository:
https://github.com/Thomas-Schatz/perceptual-tuning-pnas. We provide the original [segments.txt]() and [test.item]().

### Create directory

Set the environment variable `DATASET` to your desired path to the dataset.

```bash
mkdir -p $DATASET/raw
ln -s $CSJ_WAV $DATASET/raw/wav
ln -s $CSJ_XML $DATASET/raw/xml
```

with `$CSJ_WAV` the folder containing all the audio files, and `$CSJ_XML` the one containing all the XML files.

### Voice Activity Detection

Run this command with your own `HUGGING_FACE_HUB_TOKEN`:

```bash
plearning data vad $DATASET/raw/wav $DATASET/rttm $HUGGING_FACE_HUB_TOKEN
```

### Prepare the datasets

```bash
python prepare_csj.py $DATASET ./segments.txt ./test.item
```

### Segment the training set and create smaller subsets

```bash
plearning data segment $DATASET/csv/train_segments.csv $DATASET/raw/wav $DATASET/train/full
plearning data partition $DATASET/train/full $DATASET/csv/train_segments.csv
```

---

[1]: Thomas Schatz, Naomi H. Feldman, Sharon Goldwater, Xuan-Nga Cao and Emmanuel Dupoux, "Early phonetic learning without phonetic categories: Insights from large-scale simulations on realistic input" Proceedings of the National Academy of Sciences Feb 2021, 118 (7) e2001844118; DOI: 10.1073/pnas.2001844118
