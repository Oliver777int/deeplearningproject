import numpy as np
import torch
from torch.utils.data import DataLoader

num_of_input = 3
batch_size = 10
learning_rate = 0.005
number_of_epochs = 500


class CustomCsvDataset():
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset.shape)

    def __getitem__(self, idx):
        input = self.dataset[idx, 0:self.dataset.size(1) - 1]
        label = self.dataset[idx, self.dataset.size(1) - 1]
        return (input, label)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = torch.nn.Linear(num_of_input, 100)
        self.drop1 = torch.nn.Dropout(0.5) # add dropout if the model starts to overfit
        self.hid2 = torch.nn.Linear(100, 100)
        self.drop2 = torch.nn.Dropout(0.25)
        self.output = torch.nn.Linear(100, 1)

    def forward(self, x):
        z = torch.relu(self.hid1(x))
        z = self.drop1(z)
        z = torch.relu(self.hid2(z))
        z = self.drop2(z)
        z = torch.relu(self.hid2(z))
        z = self.drop2(z)
        z = torch.relu(self.hid2(z))
        z = self.drop2(z)
        z = torch.relu(self.hid2(z))
        z = self.drop2(z)
        z = torch.relu(self.hid2(z))
        z = self.drop2(z)
        z = torch.relu(self.hid2(z))
        z = self.drop2(z)
        z = torch.relu(self.hid2(z))
        z = self.drop2(z)
        z = torch.relu(self.hid2(z))
        z = self.drop2(z)
        z = torch.relu(self.hid2(z))
        z = self.drop2(z)
        z = torch.relu(self.hid2(z))
        z = self.drop2(z)
        z = torch.relu(self.hid2(z))
        z = self.drop2(z)
        z = torch.relu(self.hid2(z))
        z = self.drop2(z)
        z = torch.relu(self.hid2(z))
        z = self.drop2(z)
        z = self.output(z)
        return z


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('Running on the GPU')
    else:
        device = torch.device("cpu")
        print('Running on the CPU')

    file_path_train = r'C:\Users\User\OneDrive\Skola\KEX\deeplearningproject\train_data_file1.csv'
    file_path_validation = r'C:\Users\User\OneDrive\Skola\KEX\deeplearningproject\validation_data_file1.csv'

    training_data = np.loadtxt(file_path_train, dtype=np.float32, delimiter=",", skiprows=1)
    training_data = torch.from_numpy(all_data)
    train_data = CustomCsvDataset(training_data)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    validation_data = np.loadtxt(file_path_train, dtype=np.float32, delimiter=",", skiprows=1)
    validation_data = torch.from_numpy(all_data)
    val_data = CustomCsvDataset(validation_data)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    net = Net().to(device)
    net.train()

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(number_of_epochs):
        epoch_loss = 0.0

        for (batch_idx, batch) in enumerate(train_dataloader):
            X = batch[0]
            Y = batch[1]

            optimizer.zero_grad()
            output = net(X)

            loss_val = loss_func(output, Y)
            epoch_loss += loss_val.item()
            loss_val.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f'epoch {epoch} loss = {epoch_loss}')


if __name__ == '__main__':
    main()
