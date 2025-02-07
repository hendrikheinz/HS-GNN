import copy
import os.path
import random
import dgl
import numpy as np
import torch
from arguments import paser
from utilities import *
from timeit import default_timer as timer
from model import HSGNN



def training(model, train_graphs, train_labels, test_graphs, test_labels, savepath, args):
    # HSGNN train logic,
    # for simplicity, no cross validation here, but support early stopping

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    lossfunc = loss_rmse

    epoch_idx = list(range(train_labels.shape[0]))

    if len(epoch_idx) % args.bs == 0:
        num_batch = len(epoch_idx) // args.bs
    else:
        num_batch = len(epoch_idx) // args.bs + 1
    results = []

    for e in range(args.epoch):
        model.train()

        random.shuffle(epoch_idx)

        for batch in range(num_batch):
            optimizer.zero_grad()

            batch_idx = epoch_idx[batch * args.bs: min((batch + 1) * args.bs, len(epoch_idx))]
            batch_graphs = dgl.batch([train_graphs[k] for k in batch_idx])
            batch_labels = train_labels[batch_idx]

            bpreds = model(batch_graphs)

            bloss = lossfunc(bpreds, batch_labels)
            bloss.backward()

            optimizer.step()

        if e % args.valid == 0 or e == args.epoch - 1:
            model.eval()

            train_mre, train_rmrse = evaluate(model, train_graphs, train_labels, args.bs)
            test_mre, test_rmrse = evaluate(model, test_graphs, test_labels, args.bs)


            results.append([e, train_mre, train_rmrse, test_mre, test_rmrse])

            torch.save(model.state_dict(), os.path.join(savepath, 'eppoch_' + str(e) + '.pt'))

            if train_mre <= np.array(results)[:, 1].min():
                torch.save(model.state_dict(), os.path.join(savepath, 'best' + '.pt'))

            if args.echo:
                print('Epoch {:4d} / {:4d} :'.format(e, args.epoch))
                print('Train ARE {:.2f} | RMRSE {:.2f} '.format(train_mre * 100, train_rmrse * 100))
                print('Test ARE {:.2f} | RMRSE {:.2f} '.format(test_mre * 100, test_rmrse * 100))

    results = np.array(results)
    best = results[np.argmin(results[:, 1])]

    if args.echo:
        print('Best epoch: {:4d}'.format(e))
        print('Best Test MRE {:.2f} | RMRSE {:.2f}'.format(best[-2]*100, best[-1]*100))

    np.save(os.path.join(savepath, 'result.npy'), results)
    return best


def evaluate(model, graphs, labels, batchsize, pred=False):
    model.eval()
    labels = labels.cpu()

    if len(graphs) > batchsize:
        eval_index = list(range(len(graphs)))

        if len(graphs) % batchsize == 0:
            num_batches = len(graphs) // batchsize
        else:
            num_batches = len(graphs) // batchsize + 1

        preds = []
        for b in range(num_batches):
            batch_idx = eval_index[b*batchsize : min((b+1)*batchsize, len(graphs))]
            batch_graph = dgl.batch([graphs[i] for i in batch_idx])
            batch_preds = model(batch_graph).cpu().detach().view(len(batch_idx), -1)
            preds.append(batch_preds)
        preds = torch.cat(preds, dim=0)
    else:
        batch_graph = dgl.batch(graphs)
        preds = model(batch_graph).cpu().detach().view(len(graphs), -1)

    mre = loss_mre(preds, labels).item()
    rmrse = loss_rmrse(preds, labels).item()

    if pred:
        return mre, rmrse, preds.numpy()

    return  mre, rmrse



if __name__=="__main__":

    args = paser.parse_args()

    print(f"The process id is: {os.getpid()}")

    if args.device=="cpu":
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cuda:" + str(args.device))

    if args.predict == 's':
        labidx = 0
        print("Predicting strength...")
    elif args.predict == 'm':
        labidx = 1
        print("Predicting modulus...")
    else:
        raise KeyError('Unsupported prediction target, only s or m!')

    train_graphs, test_graphs, train_labels, test_labels = load_dataset(name='bundle', device=args.device, labidx=labidx)

    print(f"Train_graphs: {len(train_graphs)} | test_graphs: {len(test_graphs)} | Train_labels: {len(train_labels)} | test_labels = {len(test_labels)}")


    #######################################################################
    #
    # MODIFY 'savedir' TO THE FULL PATH WHERE HSGNN RESULTS SHOULD BE SAVED
    #
    #######################################################################
    savedir = r'path\to\save\results\exp'+args.predict
    
    
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    model = HSGNN(
        edge_types=['B','D'],
        in_dim=args.num_inp,
        hid_dim=args.num_hid,
        out_dim=1,
        num_gin=args.p,
        num_gat=args.q,
        pimg_dim=args.pimg_dim
    ).to(args.device)
    start_time = timer()
    result = training(model, train_graphs, train_labels, test_graphs, test_labels, savedir, args)
    end_time = timer()
    print(f"[INFO] Total training time (seconds): {end_time-start_time:.4f}")
    mre, rmrse, preds = evaluate(model, test_graphs, test_labels, 1, pred=True) 
