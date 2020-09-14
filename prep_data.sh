#!/bin/bash


##should use python2

stage=-1

TOTAL_SPLIT=40
DATA_ROOT=/data07/wanqiu/m5data

#prepare wav.scp for each data set. Set your $DATA_ROOT variable before run. 
if [ $stage -le 0 ]; then
for data in test dev; do
  awk -v x=$DATA_ROOT -v y=$data '{print $1,x"/"y"/"$1".wav"}' data/${data}/utt2lang > data/${data}/wav.scp
done

for data in train; do
  awk -v x=$DATA_ROOT -v y=$data '{print $1,x"/"y"/"$2"/"$1".wav"}' data/${data}/utt2lang > data/${data}/wav.scp
done
fi

# data preparation
if [ $stage -le 1 ]; then
# Shuffle (for training)
  for data in train dev test; do
    python scripts/shuffle_data.py data/${data} data/${data}_shuffle
  done
# Split wavs for parallel jobs
  for data in train_shuffle dev_shuffle; do
    python scripts/split_data.py data/${data} $TOTAL_SPLIT
  done
fi


