# Serenade: A Singing Style Conversion Framework Based on Audio Infilling

## Before you use this repo and pretrained models

### License
Commercial use is NOT allowed. Please read the [LICENSE](LICENSE) file. This repo and the models are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

### Conditions
Using this repository and the models to impersonate any singer without their consent is strictly PROHIBITED. Please use this repo and the pretrained models responsibly.

## Recipe structure
### Recipe overview
The training recipe consists of three main parts:

1. Initial model training
   - Stage 0: Data preparation, creation of train/dev/test wav.scp files
   - Stage 1: Data processing 
   - Stage 2: Dataset statistics calculation
   - Stage 3: Model training
   - Stage 4: Model inference

2. Cyclic training
   - Stage 5: Creation of converted training data, reference styles are randomly selected from the `dump` directory.
   - Stage 6: Feature extraction of converted training data 
   - Stage 7: Cyclic training, initialized from initial model
   - Stage 8: Model inference

3. Vocoder post-processing
   - Stage 9: SiFiGAN post-processing

## Usage

### Quick start
#### Pretrained Models
Download from the Google Drive link [here](https://drive.google.com/file/d/1ZhJgLHzwduELL2rzleOGDxLu4ivJ6ss-/view?usp=sharing). Then, unzip and place the directory in the `egs/gtsinger/ssc1` directory.

You can also use the script below.
```bash
./utils/download_from_google_drive.sh https://drive.google.com/open?id=1ZhJgLHzwduELL2rzleOGDxLu4ivJ6ss- .
```

#### Inference
Run the inference script with the pretrained model.

```bash
./run.sh --stage 8 --checkpoint pt_models/train-gtsigner-cyclic-sifigan/checkpoint-200000steps.pkl
```

Results will be saved in `pt_models/train-gtsigner-cyclic-sifigan/results/checkpoint-200000steps`.


### Other necessary files
- `conf/refstyles.json`: Reference styles for the conversion, you can manually set reference styles from the `dump` directory.
- `conf/serenade.yaml`: Configuration for initial model training.
- `conf/serenade_cyclic.yaml`: Configuration for cyclic training.


### Data preprocessing
Due to the license of the dataset, we cannot provide a script for downloading the data. Please download the data from [the project page](https://github.com/AaronZ345/GTSinger).

Then, specify the path to the dataset in the `run.sh` script.

```
db_root=/path/to/GTSinger
```

### Training
The README instructions are a work in progress. Detailed instructions will be available by Mar. 21 at the latest. 

```bash
cd egs/gtsinger/ssc1
./run.sh
```

### Using your own data
The preprocessing steps are tailored for the GTSinger dataset. You can skip the extraction of the ground truth score labels by setting the `--skip-gtmidi` flag to `True`. (Refer to Stage 6)

## Acknowledgements
- [ESPNet](https://github.com/espnet/espnet)
- [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN/)
- [NNSVS](https://github.com/nnsvs/nnsvs)
- [seq2seq-vc](https://github.com/unilight/seq2seq-vc)
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)
- [Phoneme-MIDI](https://github.com/seyong92/phoneme-informed-note-level-singing-transcription)

## Questions?
Please use the issues section to ask questions about the repo so that others can benefit from the answers.

## Author and Developer
**Lester Phillip Violeta**  
*Toda Laboratory, Nagoya University, Japan*  

## Advisers
**Wen-Chin Huang** [(@unilight)](https://github.com/unilight)  
*Toda Laboratory, Nagoya University, Japan*

**Tomoki Toda**  
*Toda Laboratory, Nagoya University, Japan*
