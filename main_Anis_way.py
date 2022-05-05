import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import RandomState
from pathlib import Path
import time
#test
create_new_data = False
train_data_frac = 0.8
load_model = True
save_model = True
show_histogram = False
show_val_acc = True
show_train_acc = True

num_of_input = 4
batch_size = 10000
learning_rate = 0.000002
number_of_epochs = 1000
start_time = time.time()

path_to_save_model_to = r'saved_models\model_anis5.pth'
path_to_load_from = r'saved_models\model_anis5.pth'
path_to_data = r'data\params20220425.csv'
train_path = Path(r'data\train_data.csv')
val_path = Path(r'data\val_data.csv')



if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Running on the GPU')
else:
    device = torch.device("cpu")
    print('Running on the CPU')


if create_new_data:
    df = pd.read_csv(path_to_data, names=['SWH', 'mean', 'var'])

    # Removes null values
    df.drop(df[df['SWH'].isnull()].index, inplace=True)
    rng = RandomState(4)
    df['mean'] = (df['mean'] - df['mean'].mean()) / (df['mean'].std())
    df['var'] = (df['var'] - df['var'].mean()) / (df['var'].std())
    df['mean_divided_var'] = df['mean']/df['var']
    df['mean_times_var'] = df['mean']*df['var']

    df['mean_divided_var'] = (df['mean_divided_var']-df['mean_divided_var'].mean())/(df['mean_divided_var'].std())
    df['mean_times_var'] = (df['mean_times_var'] - df['mean_times_var'].mean()) / (df['mean_times_var'].std())
    df = df[['mean', 'var', 'mean_divided_var', 'mean_times_var', 'SWH']]
    #df = df[['mean', 'var', 'SWH']]

    train = df.sample(frac=train_data_frac, random_state=rng)
    test = df.loc[~df.index.isin(train.index)]

    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(val_path, index=False)


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
        self.hid1 = torch.nn.Linear(num_of_input, 30)
        #self.drop = torch.nn.Dropout(0.25) # add dropout if the model starts to overfit
        self.hid2 = torch.nn.Linear(30, 30)
        self.hid3 = torch.nn.Linear(30, 30)
        self.output = torch.nn.Linear(30, 1)

    def forward(self, x):
        z = torch.relu(self.hid1(x))
        z = torch.relu(self.hid2(z))
        z = torch.relu(self.hid3(z))
        z = torch.relu(self.hid3(z))
        z = torch.relu(self.hid3(z))
        z = torch.relu(self.hid3(z))
        z = torch.relu(self.hid3(z))
        z = torch.relu(self.hid3(z))
        z = torch.relu(self.hid3(z))
        z = torch.relu(self.hid3(z))
        z = self.output(z)
        return z


def main():
    torch.manual_seed(4)
    np.random.seed(4)

    training_data = np.loadtxt(train_path, dtype=np.float32, delimiter=",", skiprows=1)
    if show_histogram:
        bin = np.linspace(0, 7, 100)
        plt.hist(training_data[:, 4], bins=bin)
        plt.show()

    training_data = torch.from_numpy(training_data)
    train_data = CustomCsvDataset(training_data)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    validation_data = np.loadtxt(val_path, dtype=np.float32, delimiter=",", skiprows=1)
    validation_data = torch.from_numpy(validation_data)
    val_data = CustomCsvDataset(validation_data)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    net = Net().to(device)
    # loads the old model
    if load_model:
        net.load_state_dict(torch.load(path_to_load_from))
        net.eval()

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    net.train()

    for epoch in range(number_of_epochs):
        torch.manual_seed(1+epoch) # recovery reproducibility
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

        if epoch % 1 == 0:
            print(f'epoch {epoch} loss = {epoch_loss}')

    end_time = time.time()
    print(f'time for the training: {end_time - start_time}')
    def accuracy(model, ds, ok_error):
        correct = 0
        total = 0
        for val_data, label in ds:
            with torch.no_grad():
                output = model(val_data)
                # print(output, label)
            abs_delta = np.abs(output.item()-label.item())
            #max_allow = np.abs(pct*label.item())+0.1 #added 0.1 to compensate for small waves
            if abs_delta < ok_error:
                correct += 1
            else:
                pass
                #print(f'got it wrong: val_data {val_data} label {label}, guessed {output}')
            total += 1
        acc = correct/total
        return acc

    net.eval()
    ok_error = 0.2
    if show_train_acc:
        train_acc = accuracy(net, train_data, ok_error)
        print(f'train accuracy: {train_acc}')
    if show_val_acc:
        val_acc = accuracy(net, val_data, ok_error)
        print(f'validation (0.2m) accuracy: {val_acc}')
    if show_val_acc:
        val_acc = accuracy(net, val_data, 1)
        print(f'validation (1m) accuracy: {val_acc}')
    if save_model:
        torch.save(net.state_dict(), path_to_save_model_to)

if __name__ == '__main__':
    main()
