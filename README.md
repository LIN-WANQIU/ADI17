# ADI17
## Description:
This is an end-to-end DID model based on the transformer neural network architecture.

All the experiences are carried out on the ADI17 dataset.(http://groups.csail.mit.edu/sls/downloads/adi17/) 

All the results of this experience have been summited to IALP 2020 conference. (http://www.colips.org/conferences/ialp2020/wp/)

Wanqiu Lin, Maulik Madhavi, Rohan Kumar Das and Haizhou Li, "Transformer-based Arabic Dialect Identification," International Conference on Asian Language Processing (IALP), 4-6 Dec. 2020.

## Install:
Python3 (recommend Anaconda)

PyTorch 0.4.1+

Kaldi (just for feature extraction)

## Work flow:
step 1: run prep_data.sh(for prepare data and shuffle)

step 2: run extract_feat.sh(for extract acoustic features)

step 3:run run_train.sh(for training model)

step 4:run base_line.py(for test model)
