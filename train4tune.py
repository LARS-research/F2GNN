import sys
import numpy as np
import torch
import utils
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score
from datasets import get_dataset
from model import NetworkGNN as Network

import logging
from sklearn.metrics import pairwise_distances
from torch_scatter import scatter_mean,scatter_sum
def mad_gap(x, edge_index):

    #eq 1-2 masked cos similarity
    with torch.no_grad():
        dij = torch.cosine_similarity(x[edge_index[0]], x[edge_index[1]], dim=1, eps=1e-8)
        dij = 1 - dij
        # M^{tgt} = A
        d_bar = scatter_sum(dij, edge_index[1])

    return d_bar.cpu()


def main(exp_args, run=0):
    global train_args
    train_args = exp_args

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    #np.random.seed(train_args.seed)
    torch.cuda.set_device(train_args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(train_args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(train_args.seed)

    split = True
    if train_args.data in ['Reddit', 'arxiv', 'flickr', 'arxiv_full']:
        split = False
    print('split_data:', split)
    dataset_name = train_args.data
    data, num_features, num_classes = get_dataset(dataset_name, split=split, run=run)
    # print(data.x.size(), data.edge_index.size(), num_classes, num_features)
    data = data.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    genotype = train_args.arch
    hidden_size = train_args.hidden_size

    model = Network(genotype, criterion, num_features, num_classes, hidden_size,
                    num_layers=train_args.num_layers, dropout=train_args.dropout,
                    act=train_args.activation, args=train_args)

    model = model.cuda()
    num_parameters = np.sum(np.prod(v.size()) for name, v in model.named_parameters())
    print('params size:', num_parameters)
    logging.info("genotype=%s, param size = %fMB, args=%s", genotype, utils.count_parameters_in_MB(model), train_args.__dict__)

    if train_args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            train_args.learning_rate,
            weight_decay=train_args.weight_decay
        )
    elif train_args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            train_args.learning_rate,
            momentum=train_args.momentum,
            weight_decay=train_args.weight_decay
        )
    elif train_args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            train_args.learning_rate,
            weight_decay=train_args.weight_decay
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(train_args.epochs), eta_min=train_args.min_lr)
    best_val_acc = best_test_acc = 0
    results = []
    best_line = 0
    for epoch in range(train_args.epochs):
        train_loss, train_acc, train_mad = train_trans(data, model, criterion, optimizer)
        if train_args.cos_lr:
            scheduler.step()

        valid_loss, valid_acc, val_mad, test_loss, test_acc, test_mad = infer_trans(data, model, criterion)
        results.append([valid_loss, valid_acc, val_mad, test_loss, test_acc, test_mad])

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_test_acc = test_acc
            best_line = epoch
        logging.info(
            'epoch=%s, lr=%s, train_loss=%s, train_acc=%f, valid_acc=%s, test_acc=%s, best_val_acc=%s, best_test_acc=%s, train_mad= %s, val_mad=%s, test_mad=%s',
            epoch, scheduler.get_last_lr(), train_loss, train_acc, valid_acc, test_acc, best_val_acc, best_test_acc, train_mad, val_mad, test_mad)
    print(
        'Best_results: epoch={}, val_loss={:.04f}, valid_acc={:.04f}, test_loss:{:.04f},test_acc={:.04f}, val_mad:{:.04f},test_mad:{:.04f},'.format(
            best_line, results[best_line][0], results[best_line][1], results[best_line][3], results[best_line][4], results[best_line][2], results[best_line][5]))

    return best_val_acc, best_test_acc, train_args

def train_trans(data, model, criterion, optimizer):
    model.train()
    total_loss = 0
    accuracy = 0

    # zero grad
    optimizer.zero_grad()

    # output, loss, accuracy
    mask = data.train_mask
    logits = model(data)
    madgap = mad_gap(logits, data.edge_index)

    logits = logits[mask]
    accuracy += logits.max(1)[1].eq(data.y[mask]).sum().item() / mask.sum().item()
    train_loss = criterion(logits, data.y[mask])
    total_loss += train_loss.item()
    # update w
    train_loss.backward()
    optimizer.step()

    return train_loss.item(), accuracy, madgap[mask].mean()

def infer_trans(data, model, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(data)
    preds = logits.max(1)[1]

    madgap = mad_gap(logits, data.edge_index)
    total_mad = madgap.mean()


    mask = data.val_mask.bool()
    val_loss = criterion(logits[mask], data.y[mask]).item()
    val_acc = preds[mask].eq(data.y[mask]).sum().item() / mask.sum().item()
    val_mad = madgap[mask].mean()

    mask = data.test_mask.bool()
    test_loss = criterion(logits[mask], data.y[mask]).item()
    test_acc = preds[mask].eq(data.y[mask]).sum().item() / mask.sum().item()
    test_mad = madgap[mask].mean()

    return val_loss, val_acc, val_mad, test_loss, test_acc, test_mad

if __name__ == '__main__':
    main()


