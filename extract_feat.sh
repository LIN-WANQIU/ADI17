#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --output=res_extract.txt
#SBATCH --ntasks=2

stage=2

TOTAL_SPLIT=40
DATA_ROOT=/data07/wanqiu/m5data
FEAT_FOLDER=/data07/wanqiu/m5data/fbank

. ./path.sh

if [ $stage -eq 0 ]; then
    rm -rf ${FEAT_FOLDER}/test
    mkdir -p ${FEAT_FOLDER}/test

    compute-fbank-feats --config=conf/fbank.conf \
        scp:data/test/wav.scp \
        ark,scp:data/test/feat.ark,data/test/feat.scp

    copy-feats --write-num-frames=ark,t:${FEAT_FOLDER}/test/utt2num_frames \
        ark:data/test/feat.ark \
        ark,scp:${FEAT_FOLDER}/test/feat.ark,${FEAT_FOLDER}/test/feat.scp 

    rm data/test/feat.ark data/test/feat.scp
fi

if [ $stage -eq 1 ]; then
    for data in dev_shuffle; do
        rm -rf ${FEAT_FOLDER}/${data}
        mkdir -p ${FEAT_FOLDER}/${data}
        
        # compute-fbank-feats
        for ((split=1; split<=$TOTAL_SPLIT; split++ )); do
        {
            echo ${split}
            compute-fbank-feats --config=conf/fbank.conf \
                scp:data/${data}/split40/${split}/wav.scp \
                ark,scp:data/${data}/split40/${split}/feat.ark,data/${data}/split40/${split}/feat.scp

            copy-feats --write-num-frames=ark,t:${FEAT_FOLDER}/${data}/utt2num_frames.${split} \
                ark:data/${data}/split40/${split}/feat.ark \
                ark,scp:${FEAT_FOLDER}/${data}/${split}.ark,${FEAT_FOLDER}/${data}/${split}.scp

            rm data/${data}/split40/${split}/feat.ark data/${data}/split40/${split}/feat.scp
        }&
        done 

	wait
        # concatenate the .scp files together.
        for ((split=1; split<=$TOTAL_SPLIT; split++ )); do
            cat ${FEAT_FOLDER}/${data}/${split}.scp || exit 1
        done > ${FEAT_FOLDER}/${data}/feats.scp || exit 1

        # concatenate the utt2num_frames files together.
        for ((split=1; split<=$TOTAL_SPLIT; split++ )); do
            cat ${FEAT_FOLDER}/${data}/utt2num_frames.${split} || exit 1
            rm ${FEAT_FOLDER}/${data}/utt2num_frames.${split}
        done > ${FEAT_FOLDER}/${data}/utt2num_frames || exit 1
    done
fi

#TODO : change to train
if [ $stage -eq 2 ]; then
    compute-cmvn-stats scp:${FEAT_FOLDER}/dev_shuffle/feats.scp ark,scp:${FEAT_FOLDER}/cmvn_stats.ark,${FEAT_FOLDER}/cmvn_stats.scp

    for data in dev_shuffle; do
        # compute-fbank-feats
        for ((split=1; split<=$TOTAL_SPLIT; split++ )); do
            echo ${split}
            apply-cmvn --norm-vars=true \
                scp:${FEAT_FOLDER}/cmvn_stats.scp scp:${FEAT_FOLDER}/${data}/${split}.scp \
                ark,scp:${FEAT_FOLDER}/${data}/cmvn.${split}.ark,${FEAT_FOLDER}/${data}/cmvn.${split}.scp
        done

        # concatenate the .scp files together.
        for ((split=1; split<=$TOTAL_SPLIT; split++ )); do
            cat ${FEAT_FOLDER}/${data}/cmvn.${split}.scp || exit 1
            rm ${FEAT_FOLDER}/${data}/cmvn.${split}.scp
            rm ${FEAT_FOLDER}/${data}/${split}.scp
            rm ${FEAT_FOLDER}/${data}/${split}.ark
        done > ${FEAT_FOLDER}/${data}/cmvn.scp || exit 1

        cp ${FEAT_FOLDER}/${data}/cmvn.scp data/${data}/cmvn.scp
    done
fi
