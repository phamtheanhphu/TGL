import torch
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import add_self_loops, degree

from models.TOGL.top_gnn import FiltrationGCNModel


class GCNConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConvLayer, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index=edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class TGL(torch.nn.Module):
    def __init__(self, configs, graph_dataset_batch):
        # Init parent
        super(TGL, self).__init__()
        self.configs = configs
        self.graph_dataset_batch = graph_dataset_batch

        # Topological graph neural layer
        self.topo_gnn_layer = FiltrationGCNModel(
            hidden_dim=configs.embedding_size,
            filtration_hidden=10,
            num_node_features=configs.output_size,
            num_filtrations=1,
            num_coord_funs=1,
            dim1=False
        )

        # GCN layers
        self.initial_conv = GCNConvLayer(4, configs.embedding_size)

        self.conv1 = GCNConvLayer(configs.embedding_size, configs.embedding_size)
        self.conv2 = GCNConvLayer(configs.embedding_size, configs.embedding_size)
        self.conv3 = GCNConvLayer(configs.embedding_size, configs.embedding_size)
        # self.conv4 = GCNConvLayer(configs.embedding_size, configs.embedding_size)
        # self.conv5 = GCNConvLayer(configs.embedding_size, configs.embedding_size)

        self.emb_fusion_layer = Linear(configs.embedding_size, configs.embedding_size, bias=True)

        # Output layer
        self.out = Linear(configs.embedding_size * 2, configs.output_size)

    def forward(self, x, edge_index, batch_index):
        hidden = self.initial_conv(x, edge_index)
        hidden = torch.tanh(hidden)

        hidden = self.conv1(hidden, edge_index)
        hidden = torch.tanh(hidden)

        hidden = self.conv2(hidden, edge_index)
        hidden = torch.tanh(hidden)

        hidden = self.conv3(hidden, edge_index)
        hidden = torch.tanh(hidden)

        # hidden = self.conv4(hidden, edge_index)
        # hidden = torch.tanh(hidden)
        #
        # hidden = self.conv5(hidden, edge_index)
        # hidden = torch.tanh(hidden)

        topo_hidden = self.topo_gnn_layer(self.graph_dataset_batch)
        combine_hidden = topo_hidden * hidden
        hidden = self.emb_fusion_layer(combine_hidden)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out, hidden


def training(data_loader, model):
    model.train()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.007)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for batch in data_loader:
        batch.to(device)
        optimizer.zero_grad()
        pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = torch.sqrt(loss_fn(pred, batch.y))
        loss.backward()
        optimizer.step()
    return loss, embedding


def validation(data_loader, model):
    model.eval()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.007)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for batch in data_loader:
        batch.to(device)
        optimizer.zero_grad()
        pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = torch.sqrt(loss_fn(pred, batch.y))
        loss.backward()
        optimizer.step()
    return loss, embedding
