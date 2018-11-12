# pytorch-seq2seq

This is a framework for sequence-to-sequence (seq2seq) models implemented in [PyTorch](http://pytorch.org) and made by IBM. We added a IG mode for this seq2seq mode. To use it, make sure that you have installed everything first and then type, 

```
TRAIN_PATH=data/toy_reverse/train/data.txt
DEV_PATH=data/toy_reverse/dev/data.txt
python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH  --resume --grad
```

To debug
```
python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH  --resume --grad --debug
```
# Installation
This package requires Python 2.7 or 3.6. We recommend creating a new virtual environment for this project (using virtualenv or conda).  

### Prerequisites

* Numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* PyTorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.

### Prepare toy dataset

	# Run script to generate the reverse toy dataset
    # The generated data is stored in data/toy_reverse by default
	scripts/toy.sh

### Train and play
	TRAIN_PATH=data/toy_reverse/train/data.txt
	DEV_PATH=data/toy_reverse/dev/data.txt
	# Start training
    python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH

It will take about 3 minutes to train on CPU and less than 1 minute with a Tesla K80.  Once training is complete, you will be prompted to enter a new sequence to translate and the model will print out its prediction (use ctrl-C to terminate).  Try the example below!

    Input:  1 3 5 7 9
	Expected output: 9 7 5 3 1 EOS

