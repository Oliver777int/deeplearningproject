# from torchvision.transforms import ToTensor
import numpy as np
# from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.optim as optim
# from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
from tifffile import imread
import time
from sklearn.metrics import mean_squared_error
style.use("ggplot")

# Set to true if you want to build the data
rebuild_data = False
load_model = True

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Running on the GPU')
else:
    device = torch.device("cpu")
    print('Running on the CPU')


class Build_Dataset():
    sarbilder = r'D:\CNN_storage\Balanced_dataset_sep_2021_mini'
    count = 0
    training_data = []

    def make_training_data(self):
        for f in tqdm(os.listdir(self.sarbilder)):
            try:
                path = os.path.join(self.sarbilder, f)
                img = imread(path)
                pxlimg=np.array(img)
                subpxlimg = blockshaped(pxlimg, 50, 50)     # Subdivide one 200x200 image into 16 50x50 images
                p = f.split('.tif')[0]
                label = float(p.split('_')[8])
                for i in range(16):

                    self.training_data.append([subpxlimg[i], label])
                    self.count += 1

            except Exception as e:
                print(str(e))

        np.random.shuffle(self.training_data)
        np.save('training_data.npy', self.training_data)
        print('antal sarbilder i dataset = ', self.count)


# The neural network with 3 conv and 2 fully connected layers
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5))    # Conv layer
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, (5, 5))   # Conv layer 2
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, (5, 5))  # Conv layer 3
        self.batchnorm3 = nn.BatchNorm2d(128)

        # Intermediary part that finds the value of self._to_linear
        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear, 1080)  # Fully connected layer 1
        self.fc2 = nn.Linear(1080, 1)                # Fully connected layer 2

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.batchnorm1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.batchnorm2(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = self.batchnorm3(x)

        # print(x[0].shape)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
            print(self._to_linear)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test(size=32):
    random_start = np.random.randint(len(test_x)-size)
    x, y = test_x[random_start:random_start+size], test_y[random_start:random_start+size]
    x, y = x.view(-1, 1, 50, 50).to(device), y.to(device)
    with torch.no_grad():
        val_loss = fwd_pass(x, y)
    return val_loss


def give_prediction(size=32, train_prediction=False):
    if train_prediction:
        random_start = np.random.randint(len(train_x) - size)
        x, y = train_x[random_start:random_start + size], train_y[random_start:random_start + size]
        x, y = x.view(-1, 1, 50, 50).to(device), y.to(device)
        print("Prediction on the Training set")
    else:
        random_start = np.random.randint(len(test_x) - size)
        x, y = test_x[random_start:random_start + size], test_y[random_start:random_start + size]
        x, y = x.view(-1, 1, 50, 50).to(device), y.to(device)
        print("Prediction on the validation set")
    with torch.no_grad():
        outputs = net(x)
    correct=0
    correct1=0
    correct2=0
    total=0
    for k in range(len(y)):
        if abs(outputs.detach().cpu().numpy()[k]-y.detach().cpu().numpy()[k]) < 0.1:
            correct += 1
        if abs(outputs.detach().cpu().numpy()[k] - y.detach().cpu().numpy()[k]) < 0.2:
            correct1 += 1
        if abs(outputs.detach().cpu().numpy()[k]-y.detach().cpu().numpy()[k]) < 0.5:
            correct2 += 1
        total+=1
    print("Accuracy within 10 centimeters:", round(correct / total, 3))
    print("Accuracy within 20 centmeters:", round(correct1 / total, 3))
    print("Accuracy within 50 centimeters:", round(correct2 / total, 3))

    return outputs, y


def train():
    BATCH_SIZE = 1000
    EPOCHS = 5
    with open(f'{MODEL_NAME}.log', 'a') as f:
        for epoch in range(EPOCHS):
            lossarray = []
            vallossarray = []
            for i in tqdm(range(0, len(train_x), BATCH_SIZE)):
                # print(i, i+BATCH_SIZE)
                batch_x = train_x[i:i + BATCH_SIZE].view(-1, 1, 50, 50)
                batch_y = train_y[i:i + BATCH_SIZE]
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                loss = fwd_pass(batch_x, batch_y, train=True)
                lossarray.append(loss.cpu().detach().numpy())
                if i % BATCH_SIZE/2 == 0:
                    val_loss = test(size=BATCH_SIZE)
                    vallossarray.append(val_loss.cpu().detach().numpy())
                    f.write(f'{MODEL_NAME}, {round(time.time(), 3)}, {round(float(loss), 4)}, {round(float(val_loss), 4)}\n')
            meanvalloss=np.mean(vallossarray)
            meanloss=np.mean(lossarray)
            print(f'Epoch: {epoch}. Loss: {meanloss}. Validation Loss: {meanvalloss}')


def fwd_pass(x, y, train=False):
    if train:
        net.zero_grad()
    outputs = net(x)
    #result = np.array(outputs.cpu())
    #beta = np.array(y.cpu())
    loss = loss_function(outputs, y)
    if train:
        loss.backward()
        optimizer.step()
    return loss


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    #assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    #assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))


def create_loss_graph(model_name):
    contents = open(f"{model_name}.log", "r").read().split('\n')
    times = []
    losses = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, loss, val_loss = c.split(",")

            times.append(float(timestamp))
            losses.append(float(loss))
            val_losses.append(float(val_loss))

    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

    # ax1.plot(times, accuracies, label="acc")
    # ax1.plot(times, val_accs, label="val_acc")
    ax1.legend(loc=2)
    ax2.plot(times, losses, label="loss")
    ax2.plot(times, val_losses, label="val_loss")
    ax2.legend(loc=2)
    plt.show()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("Saving Checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print("Loading checkpoint")
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def load_dataset():

    # Only rebuilds the data if rebuild_data is true
    if rebuild_data:
        make_data = Build_Dataset()
        make_data.make_training_data()

    training_data = np.load('training_data.npy', allow_pickle=True)  # Load in pixel images and labels

    # Plot the first SAR pixel image using the code below
    # plt.imshow(training_data[0][0], cmap='gray')
    # plt.show()

    # Puts data and labels in torch format
    x = torch.Tensor(np.array([i[0] for i in training_data])).view(-1, 50, 50)
    # x = (x - torch.mean(x))/torch.std(x)
    y = torch.Tensor(np.array([i[1] for i in training_data])).view(-1, 1)

    # Splits the dataset into training and testing using a random seed
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=42)
    print('training length: ', len(train_x))
    print('training label: ', len(train_y))
    print('test length: ', len(test_x))
    print('test label: ', len(test_y))
    return train_x, test_x, train_y, test_y


MODEL_NAME = f'model-{int(time.time())}'
print("the model name is ", MODEL_NAME)

train_x, test_x, train_y, test_y = load_dataset()

# Generate an instance of the Neural network
net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
loss_function = nn.MSELoss()

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))

#train()  # Trains the model
#create_loss_graph(MODEL_NAME)  # Generate loss graph

checkpoint = {'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
save_checkpoint(checkpoint)

# Plot targets on x-axis and model prediction on y-axis
output, input1 = give_prediction(1000, train_prediction=True)
output = output.cpu().detach().numpy()
input1 = input1.cpu().detach().numpy()

plt.scatter(input1, output)
plt.title("Prediction to target plot")
plt.xlabel("Real significant wave height [m]")
plt.ylabel("Predicted significant wave height [m]")
plt.show()

# Calculates mean squared error manually
# mse = mean_squared_error(input1, output)
# print(mse)

