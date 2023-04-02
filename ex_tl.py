import comet_ml

import pickle
import os
from pathlib import Path
from typing import Any, List

import numpy as np
import pytorch_lightning as pl
import torch
from ba3l.experiment import Experiment
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from sklearn import metrics
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader

from helpers.ramp import exp_warmup_linear_down

ex = Experiment("mlp_probing")


@ex.config
def default_config():
    trainer = {
        "max_epochs": 60,
        "gpus": 1,
        "num_sanity_val_steps": 0,
    }
    model = {
        "drop_out": 0.5,
        "weight_decay": 1e-3,
        "scheduler": "exp_warmup_linear_down",
        "max_lr": 1e-4,
        # cycliclr
        "base_lr": 1e-7,
        # exponential
        "warmup_epochs": 10,
        "gamma": 0.5,
    }
    data = {
        "base_dir": "embeddings/mtt/30sec/no_swa/10/",
        "metadata_dir": "mtt/",
        "batch_size": 128,
        "num_workers": 16,
        "types": "cdt",
        "reduce": "stack",
        "token_size": 768,
        "n_classes": 50,
    }

class Model(pl.LightningModule):
    @ex.capture(prefix="model")
    def __init__(
        self,
        in_features,
        n_classes,
        drop_out,
        base_lr,
        scheduler,
        max_lr,
        weight_decay,
        warmup_epochs,
        gamma,
    ):
        super().__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.gamma = gamma

        # self.bn = nn.BatchNorm1d(num_features=in_features)
        # self.linear = nn.Linear(
        #     in_features=in_features,
        #     out_features=n_classes,
        # )
        # self.relu = nn.ReLU()
        # self.bn2 = nn.BatchNorm1d(num_features=n_classes)
        # self.dropout = nn.Dropout(drop_out)
        # self.linear2 = nn.Linear(
        #     in_features=n_classes,
        #     out_features=n_classes,
        # )
        self.model = nn.Sequential(
        #   nn.BatchNorm1d(in_features),
          nn.Linear(in_features, 512),
          nn.ReLU(),
          nn.Dropout(drop_out),
        #   nn.BatchNorm1d(512),
          nn.Linear(512, n_classes),
        )
        self.sigmoid = nn.Sigmoid()

        self.best_checkpoint_path = "linear_probes/best.ckpt"

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx, key = "val"):
        x, y = batch
        y_logits = self.forward(x)
        loss = nn.functional.binary_cross_entropy_with_logits(y_logits, y)
        # Logging to TensorBoard (if installed) by default
        self.log(f"{key}_loss", loss, prog_bar=True)

        y_hat = self.sigmoid(y_logits)
        return {f"{key}_loss": loss, f"{key}_y": y, f"{key}_y_hat": y_hat}
    
    def validation_epoch_end(self, outputs: List[Any], key: str="val") -> None:
        y = torch.cat([output[f"{key}_y"] for output in outputs])
        y_hat = torch.cat([output[f"{key}_y_hat"] for output in outputs])

        y = y.detach().cpu().numpy().astype("int")
        y_hat = y_hat.detach().cpu().numpy()

        ap = metrics.average_precision_score(y, y_hat, average="macro")
        roc = metrics.roc_auc_score(y, y_hat, average="macro")

        self.log(f"{key}_ap", ap)
        self.log(f"{key}_roc", roc)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, key="test")

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.validation_epoch_end(outputs, key="test")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.max_lr, weight_decay=self.weight_decay)

        if self.scheduler == "cyclic":
            schedulers = [
                {
                "scheduler": optim.lr_scheduler.CyclicLR(
                    optimizer=optimizer,
                    base_lr=self.base_lr,
                    max_lr=self.max_lr,
                    mode="triangular2",
                    step_size_up=47,
                    cycle_momentum=False,
                ),
                "interval": "step",
                "frequency": 1,
                }
            ]

        elif self.scheduler == "exponential":
            schedulers = [
                optim.lr_scheduler.LambdaLR(
                    optimizer=optimizer,
                    lr_lambda=lambda e: (e + 1e-7) / self.warmup_epochs if e < self.warmup_epochs else 1,
                ),
                optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer,
                    gamma=self.gamma,
                    last_epoch=-1,
                ),
            ]
        
        elif self.scheduler == "exp_warmup_linear_down":
            schedulers = [
                optim.lr_scheduler.LambdaLR(
                    optimizer=optimizer,
                    lr_lambda=exp_warmup_linear_down(self.warmup_epochs, 30, 30, self.base_lr)
                ),
            ]

        return [optimizer], schedulers

    def configure_callbacks(self):
        return [
            ModelCheckpoint(
                filename = self.best_checkpoint_path,
                save_top_k=1,
                monitor="val_roc",
                mode="max",
                save_weights_only=True,
            ),
            # LearningRateMonitor(logging_interval="step")
        ]

class EmbeddingDataset(Dataset):
    @ex.capture(prefix="data")
    def __init__(self, groundtruth_file, base_dir, types, reduce):
        self.base_dir = base_dir

        self.types = types
        self.reduce = reduce

        with open(groundtruth_file, "rb") as gf:
            self.groundtruth = pickle.load(gf)

        self.filenames = {i: filename for i, filename in enumerate(list(self.groundtruth.keys()))}
        self.length = len(self.groundtruth)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        filename = self.filenames[index]
        target = self.groundtruth[filename]

        embedding_path = Path(self.base_dir, filename + ".embeddings.npy")
        embedding = np.load(embedding_path)
        embedding = self.post_process(embedding)

        return embedding, target.astype("float32")

    def post_process(self, embedding):
        # todo: implement other postprocessing
        if len(embedding.shape) == 2:
            embedding = np.mean(embedding, axis=0)

        embedding = embedding.reshape(3, -1)
        embedding_de = {
            "c": embedding[0],
            "d": embedding[1],
            "t": embedding[2],
        }
        embeddings = [v for k, v in embedding_de.items() if k in self.types]

        if self.reduce == "mean":
            return np.mean(np.array(embeddings), axis=0)
        elif self.reduce == "stack":
            return np.hstack(embeddings)


class DataModule(pl.LightningDataModule):
    @ex.capture(prefix="data")
    def __init__(
            self,
            base_dir,
            metadata_dir,
            batch_size,
            num_workers,
            types,
            reduce,
            token_size,
            n_classes,
        ):
        super().__init__()
        self.base_dir = base_dir
        self.metadata_dir = Path(metadata_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_groundtruth_file = self.metadata_dir / "groundtruth-train.pk"
        self.val_groundtruth_file = self.metadata_dir / "groundtruth-validation.pk"
        self.test_groundtruth_file = self.metadata_dir / "groundtruth-test.pk"

        self.types = types
        self.reduce = reduce
        self.token_size = token_size
        self.n_classes = n_classes

        if self.reduce == "mean":
            self.in_features = self.token_size
        elif self.reduce == "stack":
            self.in_features = self.token_size * len(self.types)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.dataset_train = EmbeddingDataset(
                groundtruth_file=self.train_groundtruth_file,
                base_dir=self.base_dir,
            )
            self.dataset_val = EmbeddingDataset(
                groundtruth_file=self.val_groundtruth_file,
                base_dir=self.base_dir,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.dataset_test = EmbeddingDataset(
                groundtruth_file=self.test_groundtruth_file,
                base_dir=self.base_dir,
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

@ex.automain
def tl_pipeline(_run, _config):
    print("starting tl experiment")

    datamodule = DataModule()
    model = Model(
        in_features=datamodule.in_features,
        n_classes=datamodule.n_classes,
    )

    trainer = ex.get_trainer()

    comet_logger = CometLogger(
        project_name=_config["project_name"],
        api_key=os.environ["COMET_API_KEY"],
        experiment_name=_run._id,
        )
    trainer.logger = comet_logger
    comet_logger.log_hyperparams(_config)

    trainer.fit(model=model, datamodule=datamodule)

    model.eval()
    trainer.test(
        model=model, datamodule=datamodule, ckpt_path=model.best_checkpoint_path
    )
