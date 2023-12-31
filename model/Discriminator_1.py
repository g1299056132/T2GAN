import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import tcn_d

class Discriminator1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer=4, dropout=0.2):
        super().__init__()

        self.tcn = tcn_d.TemporalConvNet(input_dim, [hidden_dim] * n_layer)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.n_layer = n_layer

    def forward(self, x, device='cuda'):
        #print("x1size=={}".format(x.shape))#(64,202,7)
        x = x.permute(0, 2, 1)
        #print("x2size=={}".format(x.shape))#(64,7,202)
        out = self.tcn(x).to(device)
        #print("out1size=={}".format(out.shape))#(64,128,202)         为得到（64，len,7）
        #print("out2size=={}".format(out.size(-1)))#202
        #out = self.linear(F.avg_pool1d(out, kernel_size=out.size(-1))[:, :, 0])
        out = out.permute(0, 2, 1)
        #out = out[: , : ,0:7]
        out = self.linear(out)
        output = torch.sigmoid(out)
        #print("out3size=={}".format(out.shape))#(64,128,7) 为得到（64，200,7）
        return output








    



        