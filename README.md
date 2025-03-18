# Serenade: A Singing Style Conversion Framework Based on Audio Infilling

## Before you use this repo and pretrained models

### License
Commercial use is NOT allowed. Please read the [LICENSE](LICENSE) file. This repo and the models are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

### Conditions
Using this repository and the models to impersonate any singer without their consent is strictly PROHIBITED. Please use this repo and the pretrained models responsibly.

## News and updates
- [Mar. 18] ArXiv paper [here](https://arxiv.org/abs/2503.12388) and demo page [here](https://lesterphillip.github.io/serenade_demo/).
- [Mar. 16] Initial commit.

## Usage

### Installation
```bash
conda create -n _serenade python=3.10
conda activate _serenade
pip install -e .
```

SiFiGAN is also necessary to run the recipe. A forked version of SiFiGAN is used to post-process the files.
```bash
git clone https://github.com/lesterphillip/SiFiGAN.git
cd SiFiGAN
pip install -e .
```

### Recipes
The README instructions are a work in progress. Detailed instructions will be available by Mar. 21 at the latest. 

A recipe for training a model (with pretrained models) is provided. Please refer to the README file in the recipe directory for more details.
```
cd egs/gtsinger/ssc1
./run.sh
```


## Acknowledgements
- [ESPNet](https://github.com/espnet/espnet)
- [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN/)
- [NNSVS](https://github.com/nnsvs/nnsvs)
- [seq2seq-vc](https://github.com/unilight/seq2seq-vc)
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)
- [Phoneme-MIDI](https://github.com/seyong92/phoneme-informed-note-level-singing-transcription)

## Questions?
Please use the issues section to ask questions about the repo so that others can benefit from the answers.

## Citation
Please cite the ArXiv paper if you use this repo or the pretrained models.
```bibtex
@article{violeta2025serenade,
      title={{Serenade: A Singing Style Conversion Framework Based On Audio Infilling}}, 
      author={Lester Phillip Violeta and Wen-Chin Huang and Tomoki Toda},
      year={2025},
      eprint={2503.12388},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2503.12388}, 
}
```

## Author and Developer
**Lester Phillip Violeta**  
*Toda Laboratory, Nagoya University, Japan*  

## Advisers
**Wen-Chin Huang** [(@unilight)](https://github.com/unilight)  
*Toda Laboratory, Nagoya University, Japan*

**Tomoki Toda**  
*Toda Laboratory, Nagoya University, Japan*
