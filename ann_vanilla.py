from spectral_dataset import SpectralDataset
import torch
from torch.utils.data import DataLoader
import ann_utils


class ANNVanilla:
    def __init__(self, algorithm, train_x, train_y, test_x, test_y, validation_x, validation_y, X_columns, y_column):
        self.algorithm = algorithm
        self.ann_model = None
        self.X_columns = X_columns
        self.y_column = y_column
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = validation_x.shape[1]
        ann_class = ann_utils.get_ann_by_name(algorithm)
        self.ann_model = ann_class(self.device, input_size, X_columns, y_column)
        self.model = self.ann_model
        self.model.to(self.device)
        self.train_dataset = SpectralDataset(train_x, train_y)
        self.test_dataset = SpectralDataset(test_x, test_y)
        self.validation_dataset = SpectralDataset(validation_x, validation_y)
        self.epochs = 400
        self.batch_size = 1000

    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0.001)
        n_batches = int(self.train_dataset.size()/self.batch_size) + 1
        batch_number = 0
        loss = None
        dataloader = DataLoader(self.train_dataset, persistent_workers=True, num_workers=11, batch_size=self.batch_size)
        for epoch in range(self.epochs):
            batch_number = 0
            for (x, y) in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat, additional, loss = self.model(x, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_number += 1
                #print(f'Epoch:{epoch + 1} (of {self.epochs}), Batch: {batch_number} of {n_batches}, Loss:{loss.item():.6f}')

        #torch.save(self.model, "ann.pt")]
        if hasattr(self.model, "L"):
            print(self.model.L)


    def test(self):
        batch_size = 30000
        dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()

        for (x, y) in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat, additional, loss = self.model(x, y)
            y_hat = y_hat.reshape(-1)
            y_hat = y_hat.detach().cpu().numpy()
            return y_hat

