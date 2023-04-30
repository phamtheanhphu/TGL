from torch_geometric.data import DataLoader
from typing import Callable, Optional
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_sparse import SparseTensor, cat
from sklearn.preprocessing import MinMaxScaler


class COVIDGraphDataset(InMemoryDataset):
    def __init__(self,
                 dataset_file_path,
                 relationship_file_path,
                 transform: Optional[Callable] = None):
        super().__init__('.', transform)

        self.df = pd.read_csv(dataset_file_path, encoding='utf-8')
        self.graph_df_r = pd.read_csv(relationship_file_path, header=None)

        edge_list = self.graph_df_r.values.tolist()
        nx_edges = []
        for idx, src_edge_list in enumerate(edge_list[0]):
            nx_edges.append((src_edge_list, edge_list[1][idx]))
        G = nx.DiGraph()
        G.add_edges_from(nx_edges)

        adj_matrix = nx.adjacency_matrix(G)

        node_embs = adj_matrix.todense()

        x = torch.FloatTensor(node_embs)

        if hasattr(nx, 'to_scipy_sparse_array'):
            adj = nx.to_scipy_sparse_array(G).tocoo()
        else:
            adj = nx.to_scipy_sparse_matrix(G).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        y = torch.tensor([np.zeros(x.size()[0])], dtype=torch.long)
        train_mask = torch.zeros(y.size(0), dtype=torch.bool)
        for i in range(int(y.max()) + 1):
            train_mask[(y == i).nonzero(as_tuple=False)[0]] = True

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)

        self.data, self.slices = self.collate([data])


class COVIDDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 dataset_file_path,
                 relationship_file_path,
                 configs,
                 transform=None,
                 pre_transform=None):

        self.dataset_file_path = dataset_file_path
        self.relationship_file_path = relationship_file_path
        self.configs = configs

        self.df = pd.read_csv(self.dataset_file_path, encoding='utf-8')
        self.graph_df_r = pd.read_csv(self.relationship_file_path, header=None)
        super(COVIDDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        for i in range(self.configs.window_size, self.df.shape[1]):
            x_cut = self.df.iloc[:, i - 5:i - 1]
            y_cut = self.df.iloc[:, i]
            x_list = x_cut.values.tolist()
            y_list = y_cut.values.tolist()
            x = torch.FloatTensor(x_list)
            y = torch.FloatTensor(y_list)
            edge_list = self.graph_df_r.values.tolist()
            edge_index = torch.LongTensor(edge_list)
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GraphBatchData(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (disconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(GraphBatchData, self).__init__(**kwargs)

        self.batch = batch
        self.__data_class__ = Data
        self.__slices__ = None
        self.__cumsum__ = None
        self.__cat_dims__ = None
        self.__num_nodes_list__ = None

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = GraphBatchData()
        batch.__data_class__ = data_list[0].__class__
        for key in keys + ['batch']:
            batch[key] = []

        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}
        num_nodes_list = []
        for i, data in enumerate(data_list):
            for key in keys:
                item = data[key]

                # Increase values by `cumsum` value.
                cum = cumsum[key][-1]
                if isinstance(item, Tensor) and item.dtype != torch.bool:
                    if not isinstance(cum, int) or cum != 0:
                        item = item + cum
                elif isinstance(item, SparseTensor):
                    value = item.storage.value()
                    if value is not None and value.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            value = value + cum
                        item = item.set_value(value, layout='coo')
                elif isinstance(item, (int, float)):
                    item = item + cum

                # Treat 0-dimensional tensors as 1-dimensional.
                if isinstance(item, Tensor) and item.dim() == 0:
                    item = item.unsqueeze(0)

                batch[key].append(item)

                # Gather the size of the `cat` dimension.
                size = 1
                cat_dim = data.__cat_dim__(key, data[key])
                cat_dims[key] = cat_dim
                if isinstance(item, Tensor):
                    size = item.size(cat_dim)
                elif isinstance(item, SparseTensor):
                    size = torch.tensor(item.sizes())[torch.tensor(cat_dim)]

                slices[key].append(size + slices[key][-1])
                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                cumsum[key].append(inc + cumsum[key][-1])

                if key in follow_batch:
                    if isinstance(size, Tensor):
                        for j, size in enumerate(size.tolist()):
                            tmp = f'{key}_{j}_batch'
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                torch.full((size,), i, dtype=torch.long))
                    else:
                        tmp = f'{key}_batch'
                        batch[tmp] = [] if i == 0 else batch[tmp]
                        batch[tmp].append(
                            torch.full((size,), i, dtype=torch.long))

            if hasattr(data, '__num_nodes__'):
                num_nodes_list.append(data.__num_nodes__)
            else:
                num_nodes_list.append(None)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes,), i, dtype=torch.long)
                batch.batch.append(item)

        # Fix initial slice values:
        for key in keys:
            slices[key][0] = slices[key][1] - slices[key][1]

        batch.batch = None if len(batch.batch) == 0 else batch.batch
        batch.__slices__ = slices
        batch.__cumsum__ = cumsum
        batch.__cat_dims__ = cat_dims
        batch.__num_nodes_list__ = num_nodes_list

        ref_data = data_list[0]
        for key in batch.keys:
            items = batch[key]
            item = items[0]
            if isinstance(item, Tensor):
                batch[key] = torch.cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, SparseTensor):
                batch[key] = cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, (int, float)):
                batch[key] = torch.tensor(items)

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()

    def to_data_list(self):
        r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able to reconstruct the initial objects."""

        if self.__slices__ is None:
            raise RuntimeError(
                ('Cannot reconstruct data list from batch because the batch '
                 'object was not created using `Batch.from_data_list()`.'))

        data_list = []
        for i in range(len(list(self.__slices__.values())[0]) - 1):
            data = self.__data_class__()

            for key in self.__slices__.keys():
                item = self[key]
                # Narrow the item based on the values in `__slices__`.
                if isinstance(item, Tensor):
                    dim = self.__cat_dims__[key]
                    start = self.__slices__[key][i]
                    end = self.__slices__[key][i + 1]
                    item = item.narrow(dim, start, end - start)
                elif isinstance(item, SparseTensor):
                    for j, dim in enumerate(self.__cat_dims__[key]):
                        start = self.__slices__[key][i][j].item()
                        end = self.__slices__[key][i + 1][j].item()
                        item = item.narrow(dim, start, end - start)
                else:
                    item = item[self.__slices__[key][i]:self.
                        __slices__[key][i + 1]]
                    item = item[0] if len(item) == 1 else item

                # Decrease its value by `cumsum` value:
                cum = self.__cumsum__[key][i]
                if isinstance(item, Tensor):
                    if not isinstance(cum, int) or cum != 0:
                        item = item - cum
                elif isinstance(item, SparseTensor):
                    value = item.storage.value()
                    if value is not None and value.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            value = value - cum
                        item = item.set_value(value, layout='coo')
                elif isinstance(item, (int, float)):
                    item = item - cum

                data[key] = item

            if self.__num_nodes_list__[i] is not None:
                data.num_nodes = self.__num_nodes_list__[i]

            data_list.append(data)

        return data_list

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class COVIDDataNormalizer:
    def __init__(self, original_dataset_file_path,
                 region_list_file_path,
                 train_dataset,
                 val_dataset,
                 test_dataset):
        self.original_dataset_file_path = original_dataset_file_path
        self.region_list_file_path = region_list_file_path
        self.regions = []
        self.parse_region_data()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.df = pd.read_csv(self.original_dataset_file_path, encoding='utf-8')
        self.scaler.fit(self.df.values.reshape(-1, 1))

    def get_testing_dates(self):
        testing_dates = self.df.columns.values[(len(self.train_dataset) + len(self.val_dataset) + 5):]
        return testing_dates

    def parse_region_data(self):
        with open(self.region_list_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                self.regions.append(line)

    def inverse(self, transformed_values):
        inversed_values = self.scaler.inverse_transform(transformed_values)
        non_negative_values = []
        for inversed_value in inversed_values:
            inversed_value = int(inversed_value)
            if inversed_value < 0:
                inversed_value = 0
            non_negative_values.append(inversed_value)
        return np.asarray(non_negative_values)


def data_loader(dataset, batch_size):
    data_loader_ = DataLoader(dataset, batch_size, shuffle=True)
    return data_loader_
