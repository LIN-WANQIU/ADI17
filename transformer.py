import torch
import torch.nn as nn

from encoder import Encoder

import torch.nn.functional as F

class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.m1 = nn.BatchNorm1d(17)
        self.fc1 = nn.Linear(in_features=1024,out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=17)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, padded_input, input_lengths):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        encoder_out, *_ = self.encoder(padded_input, input_lengths)
        
        input_lengths = input_lengths.reshape([-1,1,1])
        mean = torch.sum(encoder_out,1,keepdim=True)/input_lengths
        var = torch.sqrt(torch.sum(torch.square(encoder_out-mean),1,keepdim=True)/input_lengths)
        res1 =torch.cat([mean,var],2)
        res= torch.squeeze(res1,axis=1)
        
        res = res.view(-1,1024)
        res=self.fc1(res)
        res = F.relu(res)
        res=self.fc2(res)
        res = F.relu(res)
        res = self.fc3(res)
        res = self.m1(res)
        return res

    def recognize(self, input, input_length, char_list, args):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        encoder_outputs, *_ = self.encoder(input.unsqueeze(0), input_length)

        return encoder_outputs
