import kaldi_io
import numpy as np
import torch
import argparse

from encoder import Encoder
from transformer import Transformer
import torch.nn as nn
import torch.optim as optim

from m5data import m5Dataset,m5DataLoader

TRAINED_MODEL = '/home/wanqiu/final_adi/model/model_ep30.pth'

print('loading data...')
test_data = m5Dataset('data/dev_shuffle/')
test_dataloader = m5DataLoader(test_data,batch_size=10)
print('loading model...')
model = torch.load(TRAINED_MODEL )
model.cuda()

correct = 0
total = 0
model.eval()
for step,data in enumerate(test_dataloader):
    xs_pad, ilens, ys = data
    xs_pad = xs_pad.cuda()
    ilens = ilens.cuda()
    ys = ys.cuda()
    res = model(xs_pad,ilens)

    _, pred = torch.max(res, 1)      
    correct += (pred == ys).sum().item()
    total += ys.size(0)
    if step % 100 == 99:
        print ('step :',step)
accuracy = float(correct) / total
print('Acc = {:.5f}'.format(accuracy))

