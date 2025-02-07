import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import GINConv, GATConv

class HSGNN(nn.Module):

    def __init__(self,
                 edge_types,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_gin,
                 num_gat,
                 pimg_dim
                 ):
        super().__init__()

        self.botEncoder = BottomEncoder(
            edge_types=edge_types,
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_gat=num_gat,
            num_gin=num_gin,
            init_eps_gin=0.1,
            learn_eps_gin=True,
            gin_agg='mean',
            num_heads_bt=4
        )

        self.H2 = nn.ModuleList()

        num_bt_output = len(edge_types)

        for layer_h2 in range(2): # h2 layers
            fcn = MLP(
                num_layers=1,
                in_dim=hid_dim * num_bt_output + pimg_dim if layer_h2==0 else hid_dim,
                hid_dim=hid_dim,
                out_dim=hid_dim,
                batch_norm=True
            )
            self.H2.append(
                GINConv(
                    apply_func=fcn,
                    aggregator_type='mean',
                    init_eps=0.1,
                    learn_eps=True,
                    activation=nn.ReLU()
                )
            )

        fcl3 = MLP(
            num_layers=2,
            in_dim=hid_dim,
            hid_dim=hid_dim,
            out_dim=hid_dim,
            batch_norm=True
        )

        self.H3 = GINConv(
            apply_func=fcl3,
            aggregator_type='mean',
            init_eps=0.1,
            learn_eps=True,
            activation=nn.ReLU()
        )

        self.input = in_dim

        self.out_layers = MLP(
            num_layers=2,
            in_dim= (2 + num_bt_output) * hid_dim,
            hid_dim=hid_dim,
            out_dim=out_dim,
            batch_norm=False
        )


    def forward(self, graph, glbfeats=None):

        with graph.local_scope():

            graph.nodes['A'].data['feats'] = graph.nodes['A'].data['feats'][:, :self.input]

            h = self.botEncoder(graph) # [N, 2 * hid]

            graph.nodes['A'].data['h'] = h

            graph.update_all(fn.copy_u('h', 'm'), fn.max('m', 'h'), etype='G1')
            h = graph.nodes['C2'].data.pop('h')

            pimg = graph.nodes['C2'].data['pimg'].view(h.size(0), -1)

            h = torch.cat([h, pimg], dim=-1)

            subgraph = graph['C2', 'I2', 'C2']
            for conv in self.H2:
                h = conv(subgraph, h)

            graph.nodes['C2'].data['h'] = h

            # H3 layer
            graph.update_all(fn.copy_u('h', 'm'), fn.max('m', 'h'), etype='G2')
            h = graph.nodes['C3'].data.pop('h') # [N, hid]

            subgraph = graph['C3', 'I3', 'C3']
            h = self.H3(subgraph, h)

            graph.nodes['C3'].data['h'] = h

            # readout
            h1 = dgl.readout_nodes(graph, 'h', op='mean', ntype='A')
            h2 = dgl.readout_nodes(graph, 'h', op='mean', ntype='C2')
            h3 = dgl.readout_nodes(graph, 'h', op='mean', ntype='C3')

            h = torch.cat([h1, h2, h3], dim=-1)

            h = self.out_layers(h)

            return h



class BottomEncoder(nn.Module):

    def __init__(self,
                 edge_types,
                 in_dim,
                 hid_dim,
                 num_gin,
                 num_gat,
                 init_eps_gin,
                 learn_eps_gin,
                 gin_agg,
                 num_heads_bt,
                 ):
        super().__init__()


        self.botEncoders = nn.ModuleList()
        self.bottom_edges = edge_types
        for edge in edge_types:
            enc = nn.ModuleList()

            for i in range(num_gat):
                enc.append(
                    GATConv(
                        in_feats=in_dim if i==0 else hid_dim,
                        out_feats=hid_dim // num_heads_bt,
                        num_heads=num_heads_bt,
                        allow_zero_in_degree=True,
                        activation=nn.ReLU()
                    )
                )

            for i in range(num_gin):
                fcn = nn.Linear(
                    in_features=hid_dim,
                    out_features=hid_dim
                )
                '''
                fcn = MLP(
                    in_dim=hid_dim,
                    hid_dim=hid_dim,
                    out_dim=hid_dim,
                    batch_norm=True,
                    num_layers=1
                )'''
                enc.append(
                    GINConv(
                        apply_func=fcn,
                        aggregator_type=gin_agg,
                        init_eps=init_eps_gin,
                        learn_eps=learn_eps_gin,
                        #activation=None,
                        activation=nn.LeakyReLU()
                    )
                )

            self.botEncoders.append(enc)

        self.linear = nn.Linear(
            in_features=hid_dim * len(edge_types),
            out_features=hid_dim
        )

        self.relu = nn.ReLU()



    def forward(self, graph):

        with graph.local_scope():

            atom_feats = graph.nodes['A'].data['feats']

            h = []

            for edge_type, encoder in zip(self.bottom_edges, self.botEncoders):
                subgraph = graph[('A', edge_type, 'A')]
                hid_rep = atom_feats
                for gnn in encoder:
                    hid_rep = self.relu(gnn(subgraph, hid_rep)).view(atom_feats.size(0), -1)
                h.append(hid_rep)

            h = torch.cat(h, dim=1)

            return self.relu(h)



class MLP(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 out_dim,
                 hid_dim=None,
                 batch_norm=True,
                 activation=nn.LeakyReLU()
                 ):

        super().__init__()
        self.linears = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        dims = [in_dim] + [hid_dim for _ in range(num_layers-1)] + [out_dim]

        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.linears.append(nn.Linear(d_in, d_out, bias=False))

        if batch_norm:
            for _ in range(num_layers-1):
                self.batch_norms.append(nn.BatchNorm1d(hid_dim))

        self.act = activation

        self.norm = batch_norm

        #self.init_params()

    def init_params(self):
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight)

    def forward(self, h,):

        h = self.linears[0](h)

        if self.norm:
            for batch_norm, linear in zip(self.batch_norms, self.linears[1:]):
                h = self.act(h)
                h = batch_norm(h)
                h = linear(h)

        else:
            for linear in self.linears[1:]:
                h = self.act(h)
                h = linear(h)

        return h
