#!/bin/bash

. ./path.sh
FEAT_FOLDER=/data07/wanqiu/m5data/fbank

rm -rf ${FEAT_FOLDER}/acc
mkdir -p ${FEAT_FOLDER}/acc

compute-fbank-feats --config=conf/fbank.conf \
        scp:data/acc/wav.scp \
        ark,scp:data/acc/feat.ark,data/acc/feat.scp

copy-feats --write-num-frames=ark,t:${FEAT_FOLDER}/acc/utt2num_frames \
	ark:data/acc/feat.ark \
	ark,scp:${FEAT_FOLDER}/acc/feat.ark,${FEAT_FOLDER}/acc/feat.scp 

#rm data/acc/feat.ark data/acc/feat.scp

compute-cmvn-stats scp:${FEAT_FOLDER}/acc/feat.scp ark,scp:${FEAT_FOLDER}/acc/cmvn_stats.ark,${FEAT_FOLDER}/acc/cmvn_stats.scp

apply-cmvn --norm-vars=true \
	scp:${FEAT_FOLDER}/acc/cmvn_stats.scp scp:${FEAT_FOLDER}/acc/feat.scp  \
	ark,scp:${FEAT_FOLDER}/acc/cmvn.ark,${FEAT_FOLDER}/acc/cmvn.scp

cp ${FEAT_FOLDER}/acc/cmvn.ark data/acc/cmvn.ark
cp ${FEAT_FOLDER}/acc/cmvn.scp data/acc/cmvn.scp
cp ${FEAT_FOLDER}/acc/utt2num_frames data/acc/utt2num_frames
