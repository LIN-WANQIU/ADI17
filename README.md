# ADI17
## Description:
This is an end-to-end DID model based on the transformer neural network architecture.

All the experiences are carried out on the ADI17 dataset. 

All the results of this experience have been summited to APIL 2020 conference.

## Install:
Python3 (recommend Anaconda)

PyTorch 0.4.1+

Kaldi (just for feature extraction)

## Work flow:
step 1: run prep_data.sh(for prepare data and shuffle)

step 2: run extract_feat.sh(for extract acoustic features)

step 3:run run_train.sh(for training model)

step 4:run base_line.py(for test model)
