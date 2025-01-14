# How to setup

## Prerequisites for GAN model

* Linux

* Python (2.7 or later)

* numpy

* scipy

* NVIDIA GPU + CUDA 8.0 + CuDNN v5.1

* TensorFlow 1.0 or later

# Getting Started
## Steps
* clone this repo:
```
git clone https://github.com/Papyson/eeg-pipeline.git
```
* cd into directory
```
cd dual-dualgan-main
```

* download data sample:
```
https://shorturl.at/7QPjC
```

* download model checkpoint
```
https://www.kaggle.com/models/ayoobammi/dual_dual_gan_music_decoding
```

* train the model（Only verification can ignore this step）:

```
python train.py
```

* test the model:

```
python test.py
```


