import math
from sklearn.metrics import mean_squared_error, r2_score
from spectral_dataset import SpectralDataset
import torch
from torch.utils.data import DataLoader
from ann import ANN
import torch.nn.functional as F


class ANNVanilla:
    def __init__(self, train_x, train_y, test_x, test_y, validation_x, validation_y, reporter):
        self.reporter = reporter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ann_model = ANN()
        self.model = self.ann_model
        self.model.to(self.device)
        self.train_dataset = SpectralDataset(train_x, train_y)
        self.test_dataset = SpectralDataset(test_x, test_y)
        self.validation_dataset = SpectralDataset(validation_x, validation_y)
        self.epochs = 400
        self.batch_size = 1000
        self.criterion = torch.nn.MSELoss(reduction='mean')

    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0.001)
        n_batches = int(self.train_dataset.size()/self.batch_size) + 1
        batch_number = 0
        loss = None
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size)
        for epoch in range(self.epochs):
            batch_number = 0
            for (x, y) in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self.model(x, y)
                loss = self.criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_number += 1
                #print(f'Epoch:{epoch + 1} (of {self.epochs}), Batch: {batch_number} of {n_batches}, Loss:{loss.item():.6f}')
            r2, rmse = self.validate()
            r2 = round(r2,5)
            rmse = round(rmse,5)
            print(f"{epoch+1}:",r2,rmse)
        #torch.save(self.model, "ann.pt")]

    def test(self):
        batch_size = 30000
        dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()

        for (x, y) in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self.model(x, y)
            y_hat = y_hat.reshape(-1)
            y_hat = y_hat.detach().cpu().numpy()
            return y_hat

    def validate(self):
        batch_size = 30000
        dataloader = DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()

        for (x, y) in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self.model(x, y)
            y_hat = y_hat.reshape(-1)
            y_hat = y_hat.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            r2 = r2_score(y, y_hat)
            rmse = math.sqrt(mean_squared_error(y, y_hat, squared=False))

            return max(r2,0), rmse
