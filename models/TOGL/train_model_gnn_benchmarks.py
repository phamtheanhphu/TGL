#!/usr/bin/env python
"""Train a model."""
import argparse
import sys

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import topognn.models as models
import topognn.data_utils as datasets

from topognn.cli_utils import str2bool

MODEL_MAP = {
    'TopoGNN': models.FiltrationGCNModel,
    'GCN': models.GCNModel,
    'LargerGCN': models.LargerGCNModel,
    'LargerTopoGNN': models.LargerTopoGNNModel,
    'SimpleTopoGNN': models.SimpleTopoGNNModel
}


# DATASET_MAP = {
#    'IMDB-BINARY': datasets.IMDB_Binary,
#    'PROTEINS': datasets.Proteins,
#    'PROTEINS_full': datasets.Proteins_full,
#    'ENZYMES': datasets.Enzymes,
#    'DD': datasets.DD,
#    'MNIST': datasets.MNIST,
#    'CIFAR10': datasets.CIFAR10,
#    'PATTERN': datasets.PATTERN,
#    'CLUSTER': datasets.CLUSTER,
#    'Necklaces': datasets.Necklaces,
#    'Cycles': datasets.Cycles
# }


def main(model_cls, dataset_cls, args):
    # Instantiate objects according to parameters
    dataset = dataset_cls(**vars(args))
    dataset.prepare_data()

    model = model_cls(
        **vars(args),
        num_node_features=dataset.node_attributes,
        num_classes=dataset.num_classes,
        task=dataset.task
    )
    print('Running with hyperparameters:')
    print(model.hparams)

    # Loggers and callbacks
    wandb_logger = WandbLogger(
        name=f"{args.model}_{args.dataset}",
        project="topo_gnn",
        entity="topo_gnn",
        log_model=True,
        tags=[args.model, args.dataset]
    )
    early_stopping_cb = EarlyStopping(monitor="val_acc", patience=100)
    checkpoint_cb = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir,
        monitor='val_acc',
        mode='max',
        verbose=True
    )

    GPU_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0
    trainer = pl.Trainer(
        gpus=-1 if GPU_AVAILABLE else None,
        logger=wandb_logger,
        log_every_n_steps=5,
        max_epochs=args.max_epochs,
        callbacks=[early_stopping_cb, checkpoint_cb]
    )
    trainer.fit(model, datamodule=dataset)
    checkpoint_path = checkpoint_cb.best_model_path
    trainer2 = pl.Trainer(logger=False)

    model = model_cls.load_from_checkpoint(
        checkpoint_path)
    val_results = trainer2.test(
        model,
        test_dataloaders=dataset.val_dataloader()
    )[0]

    val_results = {
        name.replace('test', 'best_val'): value
        for name, value in val_results.items()
    }

    test_results = trainer2.test(
        model,
        test_dataloaders=dataset.test_dataloader()
    )[0]

    for name, value in {**val_results, **test_results}.items():
        wandb_logger.experiment.summary[name] = value


class Arguments:
    def __init__(self):
        self.model = 'TopoGNN'
        self.dataset = 'DD'
        self.max_epochs = 100
        self.dummy_var = 0
        self.paired = False
        self.merged = False
        self.hidden_dim = 1
        self.filtration_hidden = 15
        self.num_filtrations = 2
        self.dim1 = False
        self.num_coord_funs = 1
        self.num_coord_funs1 = 1
        self.lr = 0.005
        self.dropout_p = 0.2
        self.set_out_dim = 32
        self.use_node_attributes = True
        self.seed = 42
        self.batch_size = 32


# parser = argparse.ArgumentParser(add_help=False)
# parser.add_argument('--model', type=str, choices=MODEL_MAP.keys())
# parser.add_argument('--dataset', type=str)
# parser.add_argument('--max_epochs', type=int, default=1000)
# parser.add_argument('--dummy_var', type=int, default=0)
# parser.add_argument("--paired", type = str2bool, default=False)
# parser.add_argument("--merged", type = str2bool, default=False)

# parser = argparse.ArgumentParser(parents=[parent])
# parser.add_argument('--hidden_dim', type=int, default=34)
# parser.add_argument('--filtration_hidden', type=int, default=15)
# parser.add_argument('--num_filtrations', type=int, default=2)
# parser.add_argument('--dim1', type=str2bool, default=False)
# parser.add_argument('--num_coord_funs', type=int, default=3)
# parser.add_argument('--num_coord_funs1', type=int, default=3)
# parser.add_argument('--lr', type=float, default=0.005)
# parser.add_argument('--dropout_p', type=int, default=0.2)
# parser.add_argument('--set_out_dim', type=int, default=32)

# parser = argparse.ArgumentParser(parents=[parent], add_help=False)
# parser.add_argument('--use_node_attributes', type=str2bool, default=True)
# parser.add_argument('--seed', type=int, default=42)
# parser.add_argument('--batch_size', type=int, default=32)

# partial_args, _ = parser.parse_known_args()
partial_args = Arguments()
partial_args.model = 'TopoGNN'
partial_args.dataset = 'DD'
partial_args.max_epochs = 1
partial_args.dummy_var = 0
partial_args.paired = False
partial_args.merged = False

model_cls = MODEL_MAP[partial_args.model]
# dataset_cls = DATASET_MAP[partial_args.dataset]
dataset_cls = datasets.get_dataset_class(**vars(partial_args))

# parser = model_cls.add_model_specific_args(parser)
# parser = dataset_cls.add_dataset_specific_args(parser)
# args = parser.parse_args()

main(model_cls, dataset_cls, partial_args)