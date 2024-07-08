# CountGD: Multi-Modal Open-World Counting

Niki Amini-Naieni, Tengda Han, & Andrew Zisserman

Official PyTorch implementation for CountGD. Details can be found in the paper, [[Paper]](https://arxiv.org/abs/2407.04619) [[Project page]](https://www.robots.ox.ac.uk/~vgg/research/countgd/).

## Try Using CountGD to Count with Text, Visual Exemplars, or Both Together Through the App [[HERE]](https://huggingface.co/spaces/nikigoli/countgd).

<img src=img/teaser.jpg width="100%"/>

## CountGD Architecture

<img src=img/architecture.jpg width="100%"/>

## Contents
* [Preparation](#preparation)
* [CountGD Inference & Pre-Trained Weights](#countgd-inference--pre-trained-weights)
* [CountGD Train](#countgd-train)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

## Preparation
### 1. Download Dataset

In our project, the FSC-147 dataset is used.
Please visit following link to download this dataset.

* [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything)

### 2. Install GCC

Install GCC. In this project, GCC 11.3 and 11.4 were tested. The following command installs GCC and other development libraries and tools required for compiling software in Ubuntu.

```
sudo apt update
sudo apt install build-essential
```

### 3. Clone Repository

```
git clone git@github.com:niki-amini-naieni/CountGD.git
```

### 4. Set Up Anaconda Environment:

The following commands will create a suitable Anaconda environment for running the CountGD training and inference procedures. To produce the results in the paper, we used [Anaconda version 2024.02-1](https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh).

```
conda create -n countgd python=3.9.19
conda activate countgd
cd CountGD
pip install -r requirements.txt
export CC=/usr/bin/gcc-11 # this ensures that gcc 11 is being used for compilation
cd models/GroundingDINO/ops
python setup.py build install
python test.py # should result in 6 lines of * True
pip install git+https://github.com/facebookresearch/segment-anything.git
cd ../../../
```

### 5. Download Pre-Trained Weights

* Make the ```checkpoints``` directory inside the ```CountGD``` repository.

  ```
  mkdir checkpoints
  ```

* Execute the following command.

  ```
  python download_bert.py
  ```

* Download the pretrained Swin-B GroundingDINO weights.

  ```
  wget -P checkpoints https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
  ```

* Download the pretrained ViT-H Segment Anything Model (SAM) weights.

  ```
  wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  ```

## CountGD Inference & Pre-Trained Weights

The model weights used in the paper can be downloaded from [Google Drive link (1.2 GB)](https://drive.google.com/file/d/1JpfsZtcGLUM0j05CpTN4kCOrmrLf_vLz/view?usp=sharing). To reproduce the results in the paper, run the following commands after activating the Anaconda environment set up in step 4 of [Preparation](#preparation). Make sure to change the directory and file names in [datasets_fsc147_val.json](https://github.com/niki-amini-naieni/CountGD/blob/main/config/datasets_fsc147_val.json) and [datasets_fsc147_test.json](https://github.com/niki-amini-naieni/CountGD/blob/main/config/datasets_fsc147_test.json) to the ones you set up in step 1 of [Preparation](#preparation). Make sure that the model file name refers to the model that you downloaded.

For the validation set (takes ~ 26 minutes on 1 RTX 3090 GPU):

```
python -u main_inference.py --output_dir ./countgd_val -c config/cfg_fsc147_val.py --eval --datasets config/datasets_fsc147_val.json --pretrain_model_path checkpoints/checkpoint_fsc147_best.pth --options text_encoder_type=checkpoints/bert-base-uncased --crop --sam_tt_norm --remove_bad_exemplar
```

For the validation set with no Segment Anything Model (SAM) test-time normalization and, hence, slightly reduced counting accuracy (takes ~ 6 minutes on 1 RTX 3090 GPU):

```
python -u main_inference.py --output_dir ./countgd_val -c config/cfg_fsc147_val.py --eval --datasets config/datasets_fsc147_val.json --pretrain_model_path checkpoints/checkpoint_fsc147_best.pth --options text_encoder_type=checkpoints/bert-base-uncased --crop --remove_bad_exemplar
```

For the test set (takes ~ 26 minutes on 1 RTX 3090 GPU):

```
python -u main_inference.py --output_dir ./countgd_test -c config/cfg_fsc147_test.py --eval --datasets config/datasets_fsc147_test.json --pretrain_model_path checkpoints/checkpoint_fsc147_best.pth --options text_encoder_type=checkpoints/bert-base-uncased --crop --sam_tt_norm --remove_bad_exemplar
```

For the test set with no Segment Anything Model (SAM) test-time normalization and, hence, slightly reduced counting accuracy (takes ~ 6 minutes on 1 RTX 3090 GPU):

```
python -u main_inference.py --output_dir ./countgd_test -c config/cfg_fsc147_test.py --eval --datasets config/datasets_fsc147_test.json --pretrain_model_path checkpoints/checkpoint_fsc147_best.pth --options text_encoder_type=checkpoints/bert-base-uncased --crop --remove_bad_exemplar
```

* Note: Inference can be further sped up by increasing the batch size for evaluation

## CountGD Train

Training code and instructions coming soon!

## Citation

```
@article{AminiNaieni24,
  author = "Amini-Naieni, N. and Han, T. and Zisserman, A.",
  title = "CountGD: Multi-Modal Open-World Counting",
  booktitle = "arXiv",
  year = "2024",
}
```

### Acknowledgements

This repository is based on the [Open-GroundingDino](https://github.com/longzw1997/Open-GroundingDino/tree/main) and uses code from the [GroundingDINO repository](https://github.com/IDEA-Research/GroundingDINO). If you have any questions about our code implementation, please contact us at [niki.amini-naieni@eng.ox.ac.uk](mailto:niki.amini-naieni@eng.ox.ac.uk).

