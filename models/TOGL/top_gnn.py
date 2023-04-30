import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch_geometric.nn import GCNConv, global_mean_pool

from models.TOGL.topo_layer import TopologyLayer


class FiltrationGCNModel(pl.LightningModule):
    """
    GCN Model with a Graph Filtration Readout function.
    """

    def __init__(self,
                 hidden_dim,
                 filtration_hidden,
                 num_node_features,
                 num_filtrations,
                 num_coord_funs,
                 dim1=False,
                 num_coord_funs1=None,
                 lr=0.001,
                 dropout_p=0.2):
        """
        num_filtrations = number of filtration functions
        num_coord_funs = number of different coordinate function
        """
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        # self.conv2 = GCNConv(hidden_dim, hidden_dim)

        coord_funs = {"Triangle_transform": num_coord_funs,
                      "Gaussian_transform": num_coord_funs,
                      "Line_transform": num_coord_funs,
                      "RationalHat_transform": num_coord_funs
                      }

        coord_funs1 = {"Triangle_transform": num_coord_funs1,
                       "Gaussian_transform": num_coord_funs1,
                       "Line_transform": num_coord_funs1,
                       "RationalHat_transform": num_coord_funs1
                       }
        self.topo1 = TopologyLayer(
            hidden_dim, hidden_dim, num_filtrations=num_filtrations,
            num_coord_funs=coord_funs, filtration_hidden=filtration_hidden,
            dim1=dim1, num_coord_funs1=coord_funs1
        )
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.pooling_fun = global_mean_pool

        # if task is Tasks.GRAPH_CLASSIFICATION:
        #     self.pooling_fun = global_mean_pool
        # elif task in [Tasks.NODE_CLASSIFICATION, Tasks.NODE_CLASSIFICATION_WEIGHTED]:
        #     if dim1:
        #         raise NotImplementedError(
        #             "We don't yet support cycles for node classification.")
        #
        #     def fake_pool(x, batch):
        #         return x
        #
        #     self.pooling_fun = fake_pool
        # else:
        #     raise RuntimeError('Unsupported task.')

        self.dim1 = dim1
        # number of extra dimension for each embedding from cycles (dim1)
        if dim1:
            cycles_dim = num_filtrations * \
                         np.array(list(coord_funs1.values())).sum()
        else:
            cycles_dim = 0

        self.loss = torch.nn.CrossEntropyLoss()

        self.num_filtrations = num_filtrations
        self.num_coord_funs = num_coord_funs

        self.accuracy = torchmetrics.functional.accuracy
        self.accuracy_val = torchmetrics.functional.accuracy
        self.accuracy_test = torchmetrics.functional.accuracy

        self.lr = lr
        self.dropout_p = dropout_p

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training = self.training)

        out_activations, graph_activations1, filtration = self.topo1(x, data)
        # print(x.size())

        x = F.relu(out_activations)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        return x
