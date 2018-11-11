# pytorch-seq2seq

This is a framework for sequence-to-sequence (seq2seq) models implemented in [PyTorch](http://pytorch.org) and made by IBM. We added a IG mode for this seq2seq mode. To use it type, 

```
TRAIN_PATH=data/toy_reverse/train/data.txt
DEV_PATH=data/toy_reverse/dev/data.txt
python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH  --resume --grad
```

# Installation
This package requires Python 2.7 or 3.6. We recommend creating a new virtual environment for this project (using virtualenv or conda).  

### Prerequisites

* Numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* PyTorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.
