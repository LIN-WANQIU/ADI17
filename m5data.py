import kaldi_io
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from utils import  pad_list

LFR_m = 4
LFR_n = 3


def load_lang2idx():
    lang2idx = {}
    lines = open('language_id_initial').readlines()
    for line in lines:
        lang = line.rstrip().split()[0]
        idx = line.rstrip().split()[1]
        lang2idx[lang]=int(idx)    
    return lang2idx

def load_utt2lang(data_dir):
    wavs = []
    langs = []
    futt2lang = open(data_dir + 'utt2lang')
    lines = futt2lang.readlines()
    for line in lines:
        kv = line.rstrip().split()
        wavs.append(kv[0])
        langs.append(kv[1])
    return wavs,langs

def load_featpos(data_dir):
    feat_pos = {}

    fscp  = open(data_dir + 'cmvn.scp')
    lines = fscp.readlines()
    for line in lines:
        kv = line.rstrip().split()
        feat_pos[kv[0]] = kv[1]
    return feat_pos

def load_utt2frames(data_dir):
    utt2frames = {}
    futt2frames = open(data_dir + 'utt2num_frames')
    lines = futt2frames.readlines()
    for line in lines:
        kv = line.rstrip().split()
        utt2frames[kv[0]] = kv[1]
    return utt2frames


class m5Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.lang2idx = load_lang2idx()

        utt2lang = load_utt2lang(self.data_dir)
        self.wavs = utt2lang[0]
        self.langs = utt2lang[1]
        self.featpos =load_featpos(self.data_dir)
        self.utt2frames = load_utt2frames(self.data_dir)

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self,index):
        wav = self.wavs[index]
        lang = self.langs[index]
        fp = self.featpos[wav]
        nf = self.utt2frames[wav]
        label = self.lang2idx[lang]
        
        sample = {'wav':wav,'label':label,'fp':fp,'nf':nf}
        return sample

class m5DataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(m5DataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = LFRCollate(LFR_m, LFR_n)


class LFRCollate(object):
    """Build this wrapper to pass arguments(LFR_m, LFR_n) to _collate_fn"""
    def __init__(self, LFR_m, LFR_n):
        self.LFR_m = LFR_m
        self.LFR_n = LFR_n

    def __call__(self, batch):
        return _collate_fn(batch, LFR_m=self.LFR_m, LFR_n=self.LFR_n)

def _collate_fn(batch, LFR_m, LFR_n):
    xs, ys = load_inputs_and_targets(batch, LFR_m, LFR_n)

    ilens = np.array([x.shape[0] for x in xs])

    # perform padding and convert to tensor
    xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0)
    ilens = torch.from_numpy(ilens)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs_pad, ilens, ys

def load_inputs_and_targets(batch, LFR_m, LFR_n):
    # From: espnet/src/asr/asr_utils.py: load_inputs_and_targets
    # load acoustic features and target sequence of token ids
    # for b in batch:
    #     print(b[1]['input'][0]['feat'])
    xs = [kaldi_io.read_mat(b['fp']) for b in batch]
    ys = [b['label'] for b in batch]

    for i,x in enumerate(xs):
        if x.shape[0] > 1000:
            xs[i] = x[:1000]
        else :
            pass
           
    if LFR_m != 1 or LFR_n != 1:
        # xs = build_LFR_features(xs, LFR_m, LFR_n)
        xs = [build_LFR_features(x, LFR_m, LFR_n) for x in xs]

    return xs, ys


def build_LFR_features(inputs, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.

    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    # LFR_inputs_batch = []
    # for inputs in inputs_batch:
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i*n:i*n+m]))
        else: # process last LFR frame
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i*n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)
    #     LFR_inputs_batch.append(np.vstack(LFR_inputs))
    # return LFR_inputs_batch

if __name__=='__main__':
    dev_data = m5Dataset('/home/wanqiu/final_dev/data/mfcc/acc/')
    dev_dataloader = m5DataLoader(dev_data,batch_size=2)
    for i,batch_data in enumerate(dev_dataloader):
        print(i)
        print(batch_data)
        exit(0)

