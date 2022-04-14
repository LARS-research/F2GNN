import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, PPI, Reddit, Coauthor, CoraFull, gnn_benchmark_dataset, Flickr, CitationFull, Amazon, Actor, CoraFull
from torch_geometric.data import NeighborSampler
from torch_geometric.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import random
import numpy as np
import torch
from torch_sparse import SparseTensor, coalesce

from torch_geometric.data import Data
path = './data/'

def gen_uniform_60_20_20_split(data):
    skf = StratifiedKFold(5, shuffle=True, random_state=12345)
    idx = [torch.from_numpy(i) for _, i in skf.split(data.y, data.y)]
    return torch.cat(idx[:3], 0), torch.cat(idx[3:4], 0), torch.cat(idx[4:], 0)

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def split_622(data):
    split = gen_uniform_60_20_20_split(data)
    data.train_mask = index_to_mask(split[0], data.num_nodes)
    data.val_mask = index_to_mask(split[1], data.num_nodes)
    data.test_mask = index_to_mask(split[2], data.num_nodes)
    return data


def get_dataset(name, split=True, run=0):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(path, name)
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]
        if split:
            data = split_622(data)
        # print('edge_index:', data.edge_index.size())

        return data, num_features, num_classes
    elif name == 'actor':
        dataset = Actor(path + 'Actor/')
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]
        print('actor run ', run)
        data.train_mask = data.train_mask[:, run]
        data.val_mask = data.val_mask[:, run]
        data.test_mask = data.test_mask[:, run]
        # print('edge_index:', data.edge_index.size())
        return data, num_features, num_classes

    elif name in ['squirrel', 'texas', 'corafull', 'chameleon', 'wisconsin', 'cornell']:

        edge_file = path + name + '/out1_graph_edges.txt'
        feature_file = path + name + '/out1_node_feature_label.txt'
        mask_file = path + name + '/' + name + '_split_0.6_0.2_'+str(run) + '.npz'

        data = open(feature_file).readlines()[1:]
        x = []
        y = []
        for i in data:
            tmp = i.rstrip().split('\t')
            y.append(int(tmp[-1]))
            tmp_x = tmp[1].split(',')
            tmp_x = [int(fi) for fi in tmp_x]
            x.append(tmp_x)

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y)

        edges = open(edge_file)
        edges = edges.readlines()
        edge_index = []
        for i in edges[1:]:
            tmp = i.rstrip()
            tmp = tmp.split('\t')
            edge_index.append([int(tmp[0]), int(tmp[1])])
            edge_index.append([int(tmp[1]), int(tmp[0])])
        # edge_index = np.array(edge_index).transpose(1, 0)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
        print('edge_index:', edge_index.size())


        # mask
        mask = np.load(mask_file)
        train_mask = torch.from_numpy(mask['train_mask.npy']).to(torch.bool)
        val_mask = torch.from_numpy(mask['val_mask.npy']).to(torch.bool)
        test_mask = torch.from_numpy(mask['test_mask.npy']).to(torch.bool)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        # print(data.x.shape, data.y.shape, data.edge_index.shape[1], edge_index.min(), edge_index.max(), y.max() + 1)
        return data, x.shape[1], int(y.max().item()) + 1

    elif name in ['CS', 'physics']:
        if name == 'CS':
            dataset = Coauthor(path + 'CoauthorCS/', 'CS')
        else:
            dataset = Coauthor(path + 'CoauthorPhysics/', 'physics')
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]
        if split:
            data = split_622(data)
        return data, num_features, num_classes

    elif name == 'DBLP':
        dataset = CitationFull(path + 'DBLP', 'dblp')
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = split_622(dataset[0])
        return data, num_features, num_classes

    elif name == 'flickr':
        dataset = Flickr(path + 'flickr')
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]
        if split:
            data = split_622(data)
        return data, num_features, num_classes

    elif name in ['Photo', 'Computer']:
        if name == 'Computer':
            dataset = Amazon(path + 'AmazonComputers', 'Computers')
        elif name == 'Photo':
            dataset = Amazon(path + 'AmazonPhoto', 'Photo')
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]
        data = split_622(data)
        return data, num_features, num_classes







