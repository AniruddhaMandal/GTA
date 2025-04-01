import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygn
from torch_geometric.graphgym.models.encoder import AtomEncoder
from utils.encoder import LinearEncoder
from Networks.gnn_heads import InductiveEdge

class Vanilla(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, hops, dropout, type, encoder=None,graph_pooling="mean", edge_encoder=None, *args, **kwargs):
        super(Vanilla, self).__init__(*args, **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.hops = hops
        self.dropout = dropout
        self.type = type
        self.graph_pooling = graph_pooling
        self.edge_encoder = edge_encoder


        if(encoder == "Atom"):
            self.input_layer = AtomEncoder(self.hidden_dim)
        else:
            self.input_layer = LinearEncoder(self.in_dim, self.hidden_dim)
        self.mp_layers = nn.ModuleList()
        for i in range(self.hops):
            if self.type == "GCN":
                self.mp_layers.append(pygn.GCNConv(self.hidden_dim,self.hidden_dim))
            if self.type == "GIN":
                nn_obj = nn.Sequential(nn.Linear(self.hidden_dim,self.hidden_dim),
                                        nn.BatchNorm1d(self.hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_dim,self.hidden_dim))
                self.mp_layers.append(pygn.GINConv(nn_obj, eps=0.1))
            if self.type == "GAT":
                self.mp_layers.append(pygn.GATConv(self.hidden_dim,self.hidden_dim))
            if self.type == "GSAGE":
                self.mp_layers.append(pygn.GraphSAGE(self.hidden_dim,self.hidden_dim,num_layers=1))
            if self.type == "RGGC":
                self.mp_layers.append(pygn.ResGatedGraphConv(self.hidden_dim, self.hidden_dim))
        if(edge_encoder != None):
            self.head = InductiveEdge(self.hidden_dim, 1, edge_decoder=self.edge_encoder)
        self.cls_layer = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, batch):
        batch = self.input_layer(batch)
        x, edge_index = batch.x, batch.edge_index
        #x = F.relu(x)
        for i in range(self.hops):
            x = self.mp_layers[i](x, edge_index)
            if i!=(self.hops-1):
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        batch.x = x
        if self.edge_encoder != None:
            y_pred = self.head(batch)
        else:
            y_pred = self.cls_layer(batch.x)
            if self.graph_pooling == "mean":
                y_pred = pygn.global_mean_pool(y_pred, batch.batch)
            elif self.graph_pooling == "max":
                y_pred = pygn.global_max_pool(y_pred, batch.batch)
            elif self.graph_pooling == "None":
                return y_pred
            else:
                raise ValueError(f"Unknown pooling({self.graph_pooling}) used")
        return y_pred