import numpy as np
import torch
from torch.utils.data import DataLoader

num_of_input = 3
batch_size = 100
learning_rate = 0.005
number_of_epochs = 4000


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Running on the GPU')
else:
    device = torch.device("cpu")
    print('Running on the CPU')


class CustomCsvDataset():
    def __init__(self, dataset):
        self.dataset = dataset.to(device)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_data = self.dataset[idx, 0:(self.dataset.size(1) - 1)]
        label = self.dataset[idx, self.dataset.size(1) - 1]
        return input_data, label


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = torch.nn.Linear(num_of_input, 20)
        self.drop1 = torch.nn.Dropout(0.25) # add dropout if the model starts to overfit
        self.hid2 = torch.nn.Linear(20, 20)
        self.drop2 = torch.nn.Dropout(0.25)
        self.output = torch.nn.Linear(20, 1)

    def forward(self, x):
        z = torch.relu(self.hid1(x))
       #z = self.drop1(z)
        z = torch.relu(self.hid2(z))
        #z = self.drop2(z)
        z = torch.relu(self.hid2(z))
        z = self.output(z)
        return z


def main():
    # file_path_train = r'C:\Users\pj2m1s\Simon\skola\deeplearningproject\train_data_file1.csv'
    # file_path_validation = r'C:\Users\pj2m1s\Simon\skola\deeplearningproject\validation_data_file1.csv'
    file_path_train = r'C:\Users\User\OneDrive\Skola\KEX\deeplearningproject\train_data_file1.csv'
    file_path_validation = r'C:\Users\User\OneDrive\Skola\KEX\deeplearningproject\validation_data_file1.csv'

    training_data = np.loadtxt(file_path_train, dtype=np.float32, delimiter=",", skiprows=1)
    training_data = torch.from_numpy(training_data)
    train_data = CustomCsvDataset(training_data)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    validation_data = np.loadtxt(file_path_train, dtype=np.float32, delimiter=",", skiprows=1)
    validation_data = torch.from_numpy(validation_data)
    val_data = CustomCsvDataset(validation_data)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    net = Net().to(device)


    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    for epoch in range(number_of_epochs):
        epoch_loss = 0.0

        for (idx, X) in enumerate(train_dataloader):
            (input_data, label) = X
            optimizer.zero_grad()
            output = net(input_data)
            output = torch.squeeze(output)
            # print(f'output {output}\n label {label}')

            loss_val = loss_func(output, label)
            epoch_loss += loss_val.item()
            loss_val.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'epoch {epoch} loss = {epoch_loss}')

    def accuracy(model, ds, pct):
        correct = 0
        total = 0
        for val_data, label in ds:
            with torch.no_grad():
                output = model(val_data)
            abs_delta = np.abs(output.item()-label.item())
            max_allow = np.abs(pct*label.item())
            if abs_delta < max_allow:
                correct +=1
            total += 1
        acc = correct/total
        return acc

    net.eval()
    ok_error = 0.001
    train_acc = accuracy(net, train_data, ok_error)
    print(train_acc)
    val_acc = accuracy(net, val_data, ok_error)
    print(val_acc)



if __name__ == '__main__':
    main()
