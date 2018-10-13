import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers=1):
        super(GCN, self).__init__()

        self.input_layer = GraphConvolution(nfeat, nhid)
        self.hidden_layers = []
        for _ in range(nlayers - 1):
            self.hidden_layers.append(GraphConvolution(nhid, nhid))
        self.output_layer = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # Pass input to input layer and apply dropout
        x = F.relu(self.input_layer(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # Pass hidden representations to each hidden layer and apply dropout
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        # Pass hidden outputs to final output layer
        x = self.output_layer(x, adj)
        return F.log_softmax(x, dim=1)
