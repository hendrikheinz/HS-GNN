import argparse

paser = argparse.ArgumentParser(description="HS-GNN")

# hyper-parameters
# change with new training / dataset
paser.add_argument("--num_inp", type=int, default=4+1, help="Number of input feature dimensions (atomic node features); default: coordinates + degree + tube diameter")
paser.add_argument("--device", type=str, default="cpu", help="Torch device, cpu or number")
paser.add_argument("--predict", type=str, default='s', help="Can be ‘m’ (modulus), ‘s’ (strength), others unsupported.")

# hyper-parameters
# a set of suggested param, can be tuned during cross-validation
paser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
paser.add_argument("--wd", type=float, default=1e-5, help="Weight decay")
paser.add_argument("--epoch", type=int, default=800, help="Training epoch")
paser.add_argument("--bs", type=int, default=128, help="Batch size")
paser.add_argument("--p", type=int, default=2, help="Number of GIN convs in L1 layer")
paser.add_argument("--q", type=int, default=2, help="Number of GAT convs in L1 layer")
paser.add_argument("--num_head_bt", type=int, default=4, help="Number of GAT attention heads in L1 layer encoder")
paser.add_argument("--num_hid", type=int, default=64, help="Number of hidden representation dimensions")
paser.add_argument("--valid", type=int, default=50, help="Number of Validation Iterations during Training")
paser.add_argument("--echo", type=bool, default=True, help="Whether echo the realtime result during training")
paser.add_argument("--pimg_dim", type=int, default=64 * 2 + 7,
                   help="At h2, we let defecte information into pimg")  # it performs better when adding pimg
