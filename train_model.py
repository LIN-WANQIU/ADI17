import kaldi_io
import numpy as np
import torch
import argparse
import os,sys

from encoder import Encoder
from transformer import Transformer
import torch.nn as nn
import torch.optim as optim

from m5data import m5Dataset,m5DataLoader

CHECK_STEP = 1000
savepoint=0
LFR_m = 4
LFR_n = 3 

D = 80
beam_size = 5
nbest = 5
defaults = dict(beam_size=beam_size,
                nbest=nbest,
                decode_max_len=0,
                d_input = D,
                LFR_m = 4,
                LFR_n = 3,
                n_layers_enc = 4,
                n_head = 8,
                d_k = 64,
                d_v = 64,
                d_model = 512,
                d_inner = 2048,
                dropout=0.1,
                pe_maxlen=5000,
                d_word_vec=512,
                n_layers_dec = 2,
                tgt_emb_prj_weight_sharing=1)
args = argparse.Namespace(**defaults)

encoder = Encoder(args.d_input * args.LFR_m, args.n_layers_enc, args.n_head,
                  args.d_k, args.d_v, args.d_model, args.d_inner,
                  dropout=args.dropout, pe_maxlen=args.pe_maxlen)

model = Transformer(encoder)

log_dir='/home/wanqiu/final_adi/model/model_ep23.pth'#load pretrain model 

if os.path.exists(log_dir):
    model = torch.load(log_dir)
    start_epoch = 23
    print('load epoch {} successfully!'.format(start_epoch))
else:
    start_epoch = 0
    print('No saved model, training from scratch!')

model.cuda()


# Cross-Entropy loss,  SGD with moment
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
#scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=50, gamma=0.8)

train_dataloader = m5DataLoader(train_data,batch_size=10)

acc_data = m5Dataset('data/acc2/')
acc_dataloader = m5DataLoader(acc_data,batch_size=4)

print('start training.')

for epoch in range(start_epoch,60):
    running_loss=0.0

    model.train()
    #scheduler.step()

    for step,data in enumerate(train_dataloader):
        xs_pad, ilens, ys = data
        xs_pad = xs_pad.cuda()
        ilens = ilens.cuda()
        ys = ys.cuda()
        res = model(xs_pad,ilens)

        optimizer.zero_grad() 
        loss = criterion(res, ys)
        loss.backward()
        optimizer.step() 
        #scheduler.step()
        # print statistics
        running_loss += loss.item()  # tensor.item() 
        if step % CHECK_STEP == CHECK_STEP-1:
            correct = 0
            total = 0
            model.eval()
            for i,accdata in enumerate(acc_dataloader):
                xs_pad, ilens, ys = accdata
                xs_pad = xs_pad.cuda()
                ilens = ilens.cuda()
                ys = ys.cuda()
                res = model(xs_pad,ilens)

                _, pred = torch.max(res, 1)      
                correct += (pred == ys).sum().item()
                total += ys.size(0)
            accuracy = float(correct) / total
            if accuracy >=0.83 and savepoint % 8 == 0 :
                torch.save(model, './model/model_ep' + str(epoch+1) +'_'+str(step+1)+'.pth')
                savepoint = savepoint+1
            print('[%d, %5d] loss: %.3f | Acc = %.4f lr =%f ' %
                  (epoch + 1, step + 1, running_loss / CHECK_STEP,accuracy,optimizer.param_groups[0]['lr']))  
            running_loss = 0.0
            model.train()
            #scheduler.step()
    torch.save(model, './model/model_ep' + str(epoch+1) +'.pth')
