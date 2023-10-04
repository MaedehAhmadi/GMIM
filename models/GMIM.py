import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn
from layers import GCN, AvgReadout, Discriminator

class GMIM(nn.Module):
    def __init__(self, n_in, n_h, activation, device):
        super(GMIM, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)
        self.b_xent = nn.BCEWithLogitsLoss(reduction='none')
               
        self.device = device
        
    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return  h_1, c 
        
    def loss(self, features, adj, un_adj, use_ps, sparse, reduction='mean'):
        
        nb_nodes = features.shape[1]
        batch_size = 1
        
        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]
    
        lbl_1 = torch.ones(batch_size, nb_nodes)
        lbl_2 = torch.zeros(batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)
    
        if torch.cuda.is_available():
            shuf_fts = shuf_fts.to(self.device)
            lbl = lbl.to(self.device)
        logits = self.forward(features, shuf_fts, adj, sparse, None, None, None) 
              
        loss_MI = nn.BCEWithLogitsLoss(reduction=reduction)(logits, lbl)
        
        return  loss_MI
