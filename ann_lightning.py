import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning_machine import LightningMachine
from spectral_dataset import SpectralDataset
import logging


class ANNLightning:
    def __init__(self, train_x, train_y, test_x, test_y, validation_x, validation_y, X_columns, y_column):
        self.X_columns = X_columns
        self.y_column = y_column
        self.model = LightningMachine()
        self.train_dataset = SpectralDataset(train_x, train_y)
        self.test_dataset = SpectralDataset(test_x, test_y)
        self.validation_dataset = SpectralDataset(validation_x, validation_y)

        self.train_loader = DataLoader(self.train_dataset)
        self.test_loader = DataLoader(self.test_dataset)
        self.valid_loader = DataLoader(self.validation_dataset)

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
        best_model = LightningMachine.load_from_checkpoint(best_checkpoint_path)
        self.trainer.test(model=best_model, dataloaders=self.test_loader)
        algorithm = best_model.model
        algorithm.eval()
        y_pred = algorithm(self.test_dataset.x)
        return y_pred.detach().numpy()
