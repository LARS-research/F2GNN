from operations import *
import torch.nn.functional as F
from model_search import process_feature
from torch_geometric.nn import LayerNorm

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
    def __init__(self, primitive, in_dim, out_dim, act, with_linear=False):
        super(NaOp, self).__init__()

        self._op = NA_OPS[primitive](in_dim, out_dim)
        self.op_linear = nn.Linear(in_dim, out_dim)
        self.act = act_map(act)
        self.with_linear = with_linear

    def forward(self, x, edge_index):
        if self.with_linear:
            return self.act(self._op(x, edge_index) + self.op_linear(x))
        else:
            return self.act(self._op(x, edge_index))


class ScOp(nn.Module):
    def __init__(self, primitive):
        super(ScOp, self).__init__()
        self._op = SC_OPS[primitive]()

    def forward(self, x):
        return self._op(x)

class LaOp(nn.Module):
    def __init__(self, primitive, hidden_size, act, num_layers=None):
        super(LaOp, self).__init__()
        self._op = FF_OPS[primitive](hidden_size, num_layers)
        self.act = act_map(act)

    def forward(self, x):
        return self.act(self._op(x))

class NetworkGNN(nn.Module):

    def __init__(self, genotype, criterion, in_dim, out_dim, hidden_size, num_layers=4, dropout=0.5, act='relu',args=None):
        super(NetworkGNN, self).__init__()
        self.genotype = genotype
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self._criterion = criterion
        ops = genotype.split('||')
        self.args = args

        # pre-process
        self.lin1 = nn.Linear(in_dim, hidden_size)

        # aggregation
        self.gnn_layers = nn.ModuleList(
            [NaOp(ops[i], hidden_size, hidden_size, act, with_linear=args.with_linear) for i in range(num_layers)])

        self.bns = torch.nn.ModuleList()
        if self.args.batch_norm:
            for i in range(num_layers):
                self.bns.append(torch.nn.BatchNorm1d(hidden_size))

        self.lns = torch.nn.ModuleList()
        if self.args.layer_norm:
            for i in range(num_layers):
                self.lns.append(LayerNorm(hidden_size, affine=True))

        # selection
        num_edges = (self.args.num_layers + 2) * (self.args.num_layers + 1) / 2
        self.num_edges = int(num_edges)
        self.skip_op = nn.ModuleList()
        for i in range(self.num_edges):
            self.skip_op.append(ScOp(ops[self.num_layers + i]))

        # fuse function
        self.fuse_funcs = nn.ModuleList()
        for i in range(self.num_layers + 1):
            self.fuse_funcs.append(LaOp(ops[-self.num_layers - 1 + i], hidden_size, 'linear', num_layers=i + 1))

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim))


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # input node
        features = []
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
                layer_input += [self.skip_op[edge_id](features[i])]

            # fuse features
            tmp_input = self.fuse_funcs[layer](layer_input)

            # aggregation
            x = self.gnn_layers[layer](tmp_input, edge_index)
            if self.args.batch_norm:
                x = self.bns[layer](x)
            if self.args.layer_norm:
                x = self.lns[layer](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # output
            features += [x]
            start += (layer + 1)

        # select features for output node
        output_features = []
        for i in range(self.num_layers + 1):
            edge_id = start + i
            output_features += [self.skip_op[edge_id](features[i])]
        output = self.fuse_funcs[-1](output_features)
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.classifier(output)
        return output


