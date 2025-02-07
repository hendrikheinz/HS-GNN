import numpy as np
from dgl.data.utils import Subset
import torch
import torch.nn.functional as F
import dgl
import os



def load_dataset(name='bundle', device='cpu', labidx=0):
    if name == 'bundle':

        datapath = 'data/cntbundle/'
        graph_files = ['init.dgl', '5k.dgl', '10k.dgl', 'equib.dgl']
        # we use intermediate states (before failure) of CNTs for data augmentation during training

        graph_datasets = []

        for filename in graph_files:
            graph_datasets.append(dgl.load_graphs(os.path.join(datapath, filename))[0])

        labels = torch.FloatTensor(np.load(os.path.join(datapath, 'labels.npy'))[:, labidx]).to(device)

        # pre-defined dataset split, template family based
        train_idx = np.load(os.path.join(datapath, 'train_idx.npy'))
        test_idx = np.load(os.path.join(datapath, 'test_idx.npy'))

        train_graphs = [graph_datasets[0][i].to(device) for i in train_idx] + \
                       [graph_datasets[1][i].to(device) for i in train_idx] + \
                       [graph_datasets[2][i].to(device) for i in train_idx] + \
                       [graph_datasets[3][i].to(device) for i in train_idx]

        train_labels = labels[list(train_idx)*4]

        test_graphs = [graph_datasets[0][i].to(device) for i in test_idx]

        test_labels = labels[test_idx]

        print('Dataset Loaded...')

        return train_graphs, test_graphs, train_labels, test_labels



def loss_mre(logits, labels):
    # mean relative error
    # 1/N * sum( |y-y'|/y )
    relative_error = torch.abs(logits.squeeze() - labels) / labels
    return torch.mean(relative_error)

def loss_mrse(logits, labels):
    # mean relative square error
    # 1/N * sum( (y-y')/y)^2 )
    rse = ((logits.squeeze() - labels) / labels)**2
    return torch.mean(rse)

def loss_rmrse(logits, labels):
    # root mean relative square error
    mrse = loss_mrse(logits, labels)
    return torch.sqrt(mrse)

def loss_rmse(logits, labels):
    # relative mean square error
    # sum( (y' - mean(y'))^2 ) / sum( (y - mean(y))^2 )
    # print(logits.shape,labels.shape)
    num = torch.mean((logits.squeeze() - labels.squeeze())**2)

    denum = torch.mean((labels.squeeze() - torch.mean(labels.squeeze()))**2)

    return num / denum


def split_data(dataset, num_train, num_test, shuffle=True, random_state=2):
    from itertools import accumulate
    num_data = len(dataset)
    assert num_train + num_test <= num_data
    lengths = [num_train, num_test]
    if shuffle:
        indices = np.random.RandomState(seed=random_state).permutation(num_data)
    else:
        indices = np.arange(num_data)

    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(accumulate(lengths), lengths)]



def collate_fn_s(batch):
    graphs, features, targets = map(list, zip(*batch))
    g = dgl.batch(graphs)
    features = torch.stack(features, dim=0)
    targets = torch.stack(targets, dim=0)[:, 0]

    return g, features, targets


def collate_fn_m(batch):
    graphs, features, targets = map(list, zip(*batch))
    g = dgl.batch(graphs)
    features = torch.stack(features, dim=0)
    targets = torch.stack(targets, dim=0)[:, 1]

    return g, features, targets


def load_data(filepath):
    graphs, features = dgl.load_graphs(filepath)


    targets = features['labels']
    feats = features['globalFeats']
    #pca = features['pca']
    #pimgs = features['pimg']

    #feats = torch.cat([glbfeats, pca], dim=1)

    print('Dataset loaded!')

    return graphs, targets, feats

def loss_fn(predictions, targets):
    mse = F.mse_loss(predictions.squeeze(), targets)
    loss = torch.sqrt(mse / torch.sum(targets ** 2))

    return loss



def loss_fn_mean(predictions, targets):
    deviation = predictions - targets

    return torch.mean((deviation / targets)**2)

def load_junction_data(filepath, labidx, shuff=True, split=0.8):
    glist, labels = dgl.load_graphs(filepath)

    if shuff:
        indices = np.random.permutation(len(glist)) # shuffles indices with random seed
    else:
        np.random.seed(42)
        indices = np.random.permutation(len(glist)) # shuffles indices with fixed seed

    split_idx = int(len(glist) * split) # splits the data into training and testing where "split" should be a fraction

    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    train_graphs = [glist[i] for i in train_idx]
    train_labels = torch.stack([labels["labels"][i] for i in train_idx]) # Converts list into torch tensor

    test_graphs = [glist[i] for i in test_idx]
    test_labels = torch.stack([labels["labels"][i] for i in test_idx]) # converts labels into torch tensor

    if labidx == 0:
        train_labels = train_labels[:,0:1].squeeze() # selects the first column of labels which we assume to be "strength"
        test_labels = test_labels[:,0:1].squeeze()
    elif labidx == 1:
        train_labels = train_labels[:,1:2].squeeze() # selects the 2nd column of labels which we assume to be "modulus"
        test_labels = test_labels[:,1:2].squeeze()
    else:
        print("A valid property (labidx) was not provided.")
        quit


    print("Data loaded!")
    return train_graphs, test_graphs, train_labels, test_labels