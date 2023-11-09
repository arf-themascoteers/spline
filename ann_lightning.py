import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning_machine import LightningMachine
from spectral_dataset import SpectralDataset
import ann_utils
import torch
import logging


class ANNLightning:
    def __init__(self, algorithm, train_x, train_y, test_x, test_y, validation_x, validation_y, X_columns, y_column):
        self.algorithm = algorithm
        self.ann_model = None
        self.X_columns = X_columns
        self.y_column = y_column
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = validation_x.shape[1]
        ann_class = ann_utils.get_ann_by_name(algorithm)
        self.ann_model = ann_class(device, input_size, X_columns, y_column)
        self.model = LightningMachine(model=self.ann_model)
        self.train_dataset = SpectralDataset(train_x, train_y)
        self.test_dataset = SpectralDataset(test_x, test_y)
        self.validation_dataset = SpectralDataset(validation_x, validation_y)

        self.train_loader = DataLoader(self.train_dataset, persistent_workers=True, num_workers=11)
        self.test_loader = DataLoader(self.test_dataset, persistent_workers=True, num_workers=11)
        self.valid_loader = DataLoader(self.validation_dataset, persistent_workers=True, num_workers=11)

        self.es_callback = EarlyStopping(monitor="val_loss", mode="min")

        self.mc_callback = ModelCheckpoint(
            dirpath='checkpoints',
            filename='best_model',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            verbose=True
        )
        pl._logger.setLevel(logging.INFO)
        self.trainer = pl.Trainer(limit_train_batches=5000, max_epochs=300, callbacks=[self.mc_callback, self.es_callback], enable_progress_bar=False)

    def train(self):
        self.trainer.fit(model=self.model, train_dataloaders=self.train_loader, val_dataloaders=self.valid_loader)

    def test(self):
        best_checkpoint_path = self.mc_callback.best_model_path
        best_model = LightningMachine.load_from_checkpoint(best_checkpoint_path, model=self.ann_model)
        self.trainer.test(model=best_model, dataloaders=self.test_loader)
        algorithm = best_model.model
        algorithm.eval()
        y_pred = algorithm(self.test_dataset.x)
        return y_pred.detach().numpy()
