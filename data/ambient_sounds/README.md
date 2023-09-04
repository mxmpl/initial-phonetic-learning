# Pretraining dataset of ambient sounds

## Data

The full dataset is available [here](https://cognitive-ml.fr/downloads/init-plearning/train/ambient_sounds.tar) (54G, md5sum: a625f3564a7697cf827d478118045c1c).

It is made of:
- 78 hours of field recordings of animals from the Animal Sound Archive [1, 2]. The occurrence data from GBIF is distributed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) license. The audio files are under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) and [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0). The metadata CSV file contains the licenses of each file.
- 422 hours from Audioset [3]. [License CC BY 4.0](https://creativecommons.org/licenses/by/4.0).

## Recreate the dataset

- The audio files were downloaded using the `animalsound.py` script after having downloaded the metadata from https://doi.org/10.15468/dl.dmckt3.
- The Audioset data was selected and preprocessed using: https://github.com/marianne-m/audioset-prepocessing.
- Data was then converted to wav files at 16kHz with `plearning data remix`.

---
[1]: GBIF.org (14 June 2023) GBIF Occurrence Download  https://doi.org/10.15468/dl.dmckt3

[2]: Frommolt, K.-H., Bardeli, R., Kurth, F., & Clausen, M. (2006). The animal sound
archive at the humboldt-university of berlin: Current activities in conservation and
improving access for bioacoustic research. Advances in Bioacoustics 2, 139–144.
https://www.ibac.info/advances-in-bioacoustics-ii#aib10

[3]: Gemmeke, J. F., Ellis, D. P. W., Freedman, D., Jansen, A., Lawrence, W., Moore, R. C., Plakal, M., & Ritter, M. (2017). Audio set: An ontology and human-labeled dataset for audio events. 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 776–780. https://doi.org/10.1109/ICASSP.2017.7952261
