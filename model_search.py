import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operations import *
from torch.autograd import Variable
from genotypes import NA_PRIMITIVES,  SC_PRIMITIVES, FF_PRIMITIVES
from torch_geometric.nn import LayerNorm, BatchNorm

def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")

class NaOp(nn.Module):

    def __init__(self, in_dim, out_dim, with_linear, op_name):
        super(NaOp, self).__init__()
        self.op = NA_OPS[op_name](in_dim, out_dim)

        self.with_linear = with_linear
        if with_linear:
            self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        mixed_res = []
        if self.with_linear:
            return F.elu(self.op(x, edge_index) + self.linear(x))
        else:
            return F.elu(self.op(x, edge_index))

class NAMixedOp(nn.Module):
    def __init__(self, in_dim, out_dim, with_linear):
        super(NAMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in NA_PRIMITIVES:
            op = NA_OPS[primitive](in_dim, out_dim)
            self._ops.append(op)

        self.with_linear = with_linear
        if with_linear:
            self.linear = torch.nn.Linear(in_dim, out_dim)
    def forward(self, x, edge_index, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            if self.with_linear:
                linear_x = self.linear(x)
                mixed_res.append(w * F.elu(op(x, edge_index) + linear_x))
            else:
                mixed_res.append(w * F.elu(op(x, edge_index)))
        return sum(mixed_res)

class ScMixedOp(nn.Module):
    def __init__(self):
        super(ScMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in SC_PRIMITIVES:
            op = SC_OPS[primitive]()
            self._ops.append(op)

    def forward(self, x, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            mixed_res.append(w * op(x))
        return sum(mixed_res)

class LaMixedOp(nn.Module):

    def __init__(self, hidden_size, num_layers=None):
        super(LaMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in FF_PRIMITIVES:
            op = FF_OPS[primitive](hidden_size, num_layers)
            self._ops.append(op)

    def forward(self, x, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            mixed_res.append(w * F.relu(op(x)))
        return sum(mixed_res)

def process_feature(features, size):
    new_feature = []
    for feature in features:
        new_feature += [feature[:size]]
    return new_feature

class Network(nn.Module):

    def __init__(self, criterion, in_dim, out_dim, hidden_size, dropout=0.5, args=None, evaluate=False):
        super(Network, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = args.num_layers
        self._criterion = criterion
        self.dropout = dropout
        self.explore_num = 0
        self.with_linear = args.with_conv_linear
        self.args = args
        self.evaluate = evaluate
        self.temp = args.temp

        #pre-process Node 0
        self.lin1 = nn.Linear(in_dim, hidden_size)

        #node aggregator op, intermediate nodes
        self.gnn_layers = nn.ModuleList()
        if self.args.search_agg: #search agg.
            for i in range(self.num_layers):
                self.gnn_layers.append(NAMixedOp(hidden_size, hidden_size, self.with_linear))

        else: #fixed agg
            if '||' in self.args.agg: #random/bayes
                aggs = self.args.agg.split('||')
            else:
                aggs = [self.args.agg] * self.num_layers

            for i in range(self.num_layers):
                self.gnn_layers.append(NaOp(hidden_size, hidden_size, self.with_linear, op_name=aggs[i]))

        #skip op
        num_edges = (self.args.num_layers + 2) * (self.args.num_layers + 1) / 2
        # if self.args.fix_io:
        #     num_edges = num_edges - 1
        self.num_edges = int(num_edges)
        self.skip_op = nn.ModuleList()
        for i in range(self.num_edges):
            self.skip_op.append(ScMixedOp())

        # fuse function in each layer.
        self.fuse_funcs = nn.ModuleList()
        for i in range(self.num_layers + 1):
            self.fuse_funcs.append(LaMixedOp(hidden_size, i + 1))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim))


        #extra ops
        self.lns = nn.ModuleList()
        if self.args.layer_norm:
            self.lns.append(LayerNorm(hidden_size, affine=True))

        self.bns = nn.ModuleList()
        if self.args.batch_norm:
            self.bns.append(BatchNorm(hidden_size))

        self._initialize_alphas()

    def _get_categ_mask(self, alpha):
        log_alpha = alpha
        u = torch.zeros_like(log_alpha).uniform_()
        softmax = torch.nn.Softmax(-1)
        one_hot = softmax((log_alpha + (-((-(u.log())).log()))) / self.temp)
        return one_hot
    def _get_softmax_temp(self, alpha):
        softmax = torch.nn.Softmax(-1)
        one_hot = softmax(alpha / self.temp)
        return one_hot
    def get_one_hot_alpha(self, alpha):
        one_hot_alpha = torch.zeros_like(alpha, device=alpha.device)
        idx = torch.argmax(alpha, dim=-1)
        for i in range(one_hot_alpha.size(0)):
            one_hot_alpha[i, idx[i]] = 1.0

        return one_hot_alpha
    def forward(self, data, single_path=False):

        if self.training:
            if self.args.algo == 'darts':
                self.sc_weights = self._get_softmax_temp(self.sc_alphas)
                self.ff_weights = self._get_softmax_temp(self.ff_alphas)
                if self.args.search_agg:
                    self.agg_weights = self._get_softmax_temp(self.agg_alphas)

            elif self.args.algo == 'snas':
                self.sc_weights = self._get_categ_mask(self.sc_alphas)
                self.ff_weights = self._get_categ_mask(self.ff_alphas)
                if self.args.search_agg:
                    self.agg_weights = self._get_categ_mask(self.agg_alphas)
        else:
            if single_path:
                self.sc_weights = self.get_one_hot_alpha(self.sc_alphas)
                self.ff_weights = self.get_one_hot_alpha(self.ff_alphas)
                if self.args.search_agg:
                    self.agg_weights = self.get_one_hot_alpha(self.agg_alphas)


        output = self.forward_model(data)
        return output
    def forward_model(self, data):
        x, edge_index = data.x, data.edge_index
        features = []

        # input node 0
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        features += [x]

        # intermediate nodes
        start = 0
        for layer in range(self.num_layers):

            # select inputs
            layer_input = []
            for i in range(layer + 1):
                edge_id = start + i
                layer_input += [self.skip_op[edge_id](features[i], self.sc_weights[edge_id])]

            # fuse features
            tmp_input = self.fuse_funcs[layer](layer_input, self.ff_weights[layer])

            # aggregation
            if self.args.search_agg: # F2GNN variant
                x = self.gnn_layers[layer](tmp_input, edge_index, self.agg_weights[layer])
            else:
                x = self.gnn_layers[layer](tmp_input, edge_index)

            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.args.layer_norm:
                x = self.bns[i](x)

            # output
            features += [x]
            start += (layer + 1)

        # select features for output node
        output_features = []
        for i in range(self.num_layers + 1):
            edge_id = start + i
            output_features += [self.skip_op[edge_id](features[i], self.sc_weights[edge_id])]

        #fuse
        output = self.fuse_funcs[-1](output_features, self.ff_weights[-1])

        #classifier
        output = self.classifier(output)
        return output

    def forward_rb(self, data, gnn):

        if self.training:
            self.sc_weights = self.sc_alphas.clone() * 0
            self.ff_weights = self.ff_alphas.clone() * 0
            self.sc_weights[[i for i in range(self.num_edges)], gnn[:self.num_edges]] = 1.0
            self.ff_weights[[i for i in range(self.num_layers+1)], gnn[self.num_edges:self.num_edges + self.args.num_layers+1]] = 1.0
            if self.args.search_agg:
                self.agg_weights = self.agg_alphas.clone() * 0
                self.agg_weights[[i for i in range(self.args.num_layers)], gnn[self.num_edges + self.args.num_layers+1:]] = 1.0

        output = self.forward_model(data)
        return output

    def _initialize_alphas(self):
        num_sc_ops = len(SC_PRIMITIVES)
        num_ff_ops = len(FF_PRIMITIVES)
        num_na_ops = len(NA_PRIMITIVES)

        if self.args.algo in ['darts', 'random', 'bayes']:
            self.sc_alphas = Variable(1e-3 * torch.randn(self.num_edges, num_sc_ops).cuda(), requires_grad=True)
            self.ff_alphas = Variable(1e-3 * torch.randn(self.args.num_layers + 1, num_ff_ops).cuda(), requires_grad=True)
            if self.args.search_agg:
                self.agg_alphas = Variable(1e-3 * torch.randn(self.args.num_layers, num_na_ops).cuda(), requires_grad=True)

        elif self.args.algo == 'snas':
            self.sc_alphas = Variable(torch.ones(self.num_edges, num_sc_ops).normal_(self.args.loc_mean, self.args.loc_std).cuda(), requires_grad=True)
            self.ff_alphas = Variable(torch.ones(self.args.num_layers + 1, num_ff_ops).normal_(self.args.loc_mean, self.args.loc_std).cuda(), requires_grad=True)
            if self.args.search_agg:
                self.agg_alphas = Variable(torch.ones(self.args.num_layers, num_na_ops).normal_(self.args.loc_mean, self.args.loc_std).cuda(),
                                           requires_grad=True)

        if self.args.search_agg:
            self._arch_parameters = [
                self.sc_alphas,
                self.ff_alphas,
                self.agg_alphas,
            ]
        else:
            self._arch_parameters = [
                self.sc_alphas,
                self.ff_alphas,
            ]

    def arch_parameters(self):
        return self._arch_parameters

    def _parse(self, sc_weights, la_weights):
        gene = []

        if '||' in self.args.agg:
            aggs = self.args.agg.split('||')
            gene.append(aggs[:])
        else:
            aggs = [self.args.agg] * self.args.num_layers
        gene += aggs

        sc_indices = torch.argmax(sc_weights, dim=-1)
        for k in sc_indices:
            gene.append(SC_PRIMITIVES[k])
        la_indices = torch.argmax(la_weights, dim=-1)
        for k in la_indices:
            gene.append(FF_PRIMITIVES[k])
        return '||'.join(gene)
    def sparse_single(self,weights, opsets):
        gene = []
        indices = torch.argmax(weights, dim=-1)
        for k in indices:
            gene.append(opsets[k])
        return gene

    def genotype(self, sample=False):

        gene = []

        # agg
        if self.args.search_agg:
            if sample:
                # random/bayes, the ops are inits in the agg_weights
                gene += self.sparse_single(F.softmax(self.agg_weights, dim=-1).data.cpu(), NA_PRIMITIVES)
            else:
                # ours, the ops are derived from the super net.
                gene += self.sparse_single(F.softmax(self.agg_alphas, dim=-1).data.cpu(), NA_PRIMITIVES)
        else:
            gene += [self.args.agg] * self.args.num_layers

        # topology
        if sample:
            gene += self.sparse_single(F.softmax(self.sc_weights, dim=-1).data.cpu(), SC_PRIMITIVES)
            gene += self.sparse_single(F.softmax(self.ff_weights, dim=-1).data.cpu(), FF_PRIMITIVES)
        else:
            gene += self.sparse_single(F.softmax(self.sc_alphas, dim=-1).data.cpu(), SC_PRIMITIVES)
            gene += self.sparse_single(F.softmax(self.ff_alphas, dim=-1).data.cpu(), FF_PRIMITIVES)

        return '||'.join(gene)

