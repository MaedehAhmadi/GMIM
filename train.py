
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
from tqdm import tqdm
import math
import sklearn
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

from models import GMIM
from utils import process
from utils.EM_utils_1 import *
from utils.EM_utils_2 import *
from new_data_loader import load_data
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0")

# Copy the files in the following link to the data folder
# https://drive.google.com/drive/folders/1rnUPsGmh2uaj0H6kJB--mxzPZEIOvglm?usp=sharing

dataset = 'flickr'

if dataset=='flickr':
    nb_clusters = 9
    use_ppr = False
    anl = 0
    MI_W = 2000
    T1 = 400
    hid_units = 256
if dataset=='pubmed':
    nb_clusters = 3
    use_ppr = True
    anl = 1
    MI_W = 2000
    T1 = 1000
    hid_units = 256
    

print("Loading {} data...".format(dataset))
adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset)

features = process.preprocess_features(features)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
arr_adj = adj.toarray()
sparse = False
adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])

if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])

if torch.cuda.is_available():
    features = features.to(device)
    if sparse:
        sp_adj = sp_adj.to(device)
    else:
        adj = adj.to(device)


if use_ppr:
    print("computing ppr...")
    ppr = torch.FloatTensor(process.compute_ppr(arr_adj)[np.newaxis]).cuda()
    del arr_adj
else:
    ppr = 0


anls = [0, 0.0044, 0.0025]

# training params
batch_size = 1
nb_epochs = 1000
patience = 40
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
nonlinearity = 'prelu'



model = GMIM(ft_size, hid_units, nonlinearity, device).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0
un_adj = 0

#------------Initialize deep params via optimizing MI-----------------#

print("Initialization...")
for epoch in range(nb_epochs):
    model.train()
    optimiser.zero_grad()

    if use_ppr:
        loss = model.loss(features, ppr, un_adj, False, sparse)
    else:
        loss = model.loss(features, sp_adj if sparse else adj, un_adj, False, sparse)


    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), "best_model_MI.pkl")
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        break

    loss.backward()
    optimiser.step()

model.load_state_dict(torch.load("best_model_MI.pkl"))

#----------------Initialize GMM params With Kmeans------------------#

if use_ppr:
    embeds, _ = model.embed(features, ppr, sparse, None)
else:
    embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)

embeds = torch.squeeze(embeds)
embeds_np = embeds.cpu()
embeds_np = embeds_np.detach().numpy()

# Kmeans on new embeddings
kmeans = KMeans(n_clusters=nb_clusters, random_state=0).fit(embeds_np)

resp = F.one_hot(torch.tensor(kmeans.labels_).to(torch.int64), num_classes=nb_clusters).float().to(device)
resp[resp==0] = -7000 #1e-10
resp[resp==1] = 0
weights, means, covs = m_step(embeds, resp)

weights = weights.to(device)
means =  means.to(device)
covs = covs.to(device)
print("Initialization done!")


#--------------Update GMM and MI params iteratively------------------#

optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

for it in tqdm(range(T1), desc ="Training progress"):

    model.train()
    optimiser.zero_grad()

    #----update GMM params (encoder params fixed)---#

    if use_ppr:
        embeds, _ = model.embed(features, ppr, sparse, None)
    else:
        embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
    embeds = torch.squeeze(embeds)

    _, log_resp = estimate_log_prob_resp(embeds, means.float(), covs.float(), 'diag', weights.float())

    weights, means, covs = m_step(embeds, log_resp)

    means = means.detach()
    covs = covs.detach()
    weights = weights.detach()

    #---- Update deep params (GMM params fixed) ----#

    if use_ppr:
        embeds, _ = model.embed(features, ppr, sparse, None)
    else:
        embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
    embeds = torch.squeeze(embeds)

    log_P_x, resp = estimate_log_prob_resp(embeds, means.float(), covs.float(), 'diag', weights) #(N,) log p(X)
    loss_logP = - torch.mean(log_P_x)

    if it==0:
        pre_resp = resp

    if use_ppr:
        loss_MI = model.loss(features, ppr, un_adj, False, sparse)
    else:
        loss_MI = model.loss(features, sp_adj if sparse else adj, un_adj, False, sparse)


    if anl>0:
        loss_logP_weight = (np.exp(anls[anl] * it) - np.exp(-anls[anl] * it)) \
            / (np.exp(anls[anl] * it) + np.exp(-anls[anl] * it))
    else:
        loss_logP_weight = 1

    loss = loss_logP_weight * loss_logP + MI_W * loss_MI

    optimiser.zero_grad()
    loss.backward(retain_graph=True)
    optimiser.step()

if use_ppr:
    embeds, _ = model.embed(features, ppr, sparse, None)
else:
    embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)

embeds = torch.squeeze(embeds)
embeds_np = embeds.cpu().detach().numpy()

kmeans = KMeans(n_clusters=nb_clusters, random_state=0).fit(embeds_np)
acc, _ = process.cluster_acc(labels, kmeans.labels_)
nmi = normalized_mutual_info_score(labels, kmeans.labels_)
ari = adjusted_rand_score(labels, kmeans.labels_)

print("ACC= {:.2f} , NMI= {:.2f} , ARI={:.2f}".format(acc*100, nmi*100, ari*100))
    

    
