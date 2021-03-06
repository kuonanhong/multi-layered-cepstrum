# multi-layered-cepstrum

Source code of the paper [Multi-Layered Cepstrum for Instantaneous Frequency Estimation](https://ieeexplore.ieee.org/document/8646684) accepted at GlobalSIP 2018.

## Requirements
* Numpy
* Scipy
* Torch (optional for faster performance)
* Librosa (file IO)
* Matplotlib (visualization)
* Pypianoroll (visualization)

## Quick Start

1. Donwload [bach10](http://music.cs.northwestern.edu/data/Bach10.html) dataset.
2. Run with default parameters.
```
python bach10.py your/download/path/01-AchGottundHerr/01-AchGottundHerr.wav \
       --f0_file your/download/path/01-AchGottundHerr/01-AchGottundHerr-GTF0s.mat
.
.
.
Time cost: 5.6769 seconds
Precision: 0.7661, Recall: 0.9170, F-score:, 0.8348
```
![](images/bach10_1.png)


## Torch

We also implement a faster version using PyTorch as `torch_bach10.py`, and it can run roughly 2 times faster on CPU.
Add `--cuda` can further utilize computational resources on GPU.

