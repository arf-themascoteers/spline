import math
from sklearn.metrics import mean_squared_error, r2_score
from spectral_dataset import SpectralDataset
import torch
from torch.utils.data import DataLoader
from ann import ANN
from reporter import Reporter


class ANNVanilla:
    def __init__(self, train_x, train_y, test_x, test_y, validation_x, validation_y):
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
        self.write_columns()
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0.001)
        n_batches = int(self.train_dataset.size()/self.batch_size) + 1
        batch_number = 0
        loss = None
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size)
        row = None
        for epoch in range(self.epochs):
            batch_number = 0
            rows = []
            for (x, y) in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_number += 1
                row = self.dump(y, y_hat, loss, epoch + 1, batch_number)
                rows.append(row)

                #print(f'Epoch:{epoch + 1} (of {self.epochs}), Batch: {batch_number} of {n_batches}, Loss:{loss.item():.6f}')
            r2, rmse = self.validate()
            r2 = round(r2,5)
            rmse = round(rmse,5)
            print("".join([str(i).ljust(10) for i in row]))
            Reporter.write_rows(rows)

        #torch.save(self.model, "ann.pt")]

    def test(self):
        batch_size = 30000
        dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()

        for (x, y) in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self.model(x)
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
            y_hat = self.model(x)
            y_hat = y_hat.reshape(-1)
            y_hat = y_hat.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            r2 = r2_score(y, y_hat)
            rmse = math.sqrt(mean_squared_error(y, y_hat))

            return max(r2,0), rmse

    def write_columns(self):
        columns = ["epoch","batch","r2","loss"]
        serial = 1
        for p in self.ann_model.get_params():
            columns.append(f"SI#{serial}")
            serial = serial+1
            for mp in p["params"]:
                columns.append(mp["name"])
        print("".join([c.ljust(10) for c in columns]))
        Reporter.write_columns(columns)

    def dump(self, y, y_hat, loss, epoch, batch_number):
        plot_items = []
        y_hat = y_hat.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        r2 = round(r2_score(y, y_hat),5)
        r2 = max(0,r2)
        plot_items.append({"name":"r2","value":r2})
        rmse = round(math.sqrt(loss.item()),5)
        plot_items.append({"name":"rmse","value":rmse})
        row = [epoch, batch_number, r2, rmse]
        serial = 1
        for p in self.ann_model.get_params():
            row.append(p["si"])
            serial = serial+1
            for mp in p["params"]:
                row.append(round(mp["value"],5))
        return row