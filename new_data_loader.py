import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import sklearn.preprocessing as preprocess

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset):
    
    # load the data: x, tx, allx, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    if dataset == 'flickr':
        features, adj, label = load_flickr()
        return adj, features, label,  0,0,0

    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
        # objects.append(
        #     pkl.load(open("data_new/ind.{}.{}".format(dataset, names[i]), 'rb')))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    #print(adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    return adj, features, np.argmax(labels, 1), 0,0,0 #idx_train, idx_val, idx_test


def load_data_flickr():
    f = np.loadtxt("data/flickr.feature", dtype = float)
    l = np.loadtxt("data/flickr.label", dtype = int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    label = np.array(l) 
    
    return features, label


def load_graph_flickr(n):
    struct_edges = np.genfromtxt("data/flickr.edge", dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(n, n), dtype=np.float32)
    ss = sp.csr_matrix(sadj)
   
    return ss 

def load_flickr():
    features, labels = load_data_flickr()
    adj = load_graph_flickr(features.shape[0])
    
    return features, adj, labels

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index








