"""
Dependencies:
torch: 0.4
matplotlib
"""
import torch
import torch.utils.data
import torch.nn.functional as F
#import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        #self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        #self.hidden3 = torch.nn.Linear(n_hidden*2, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        #x = F.relu(self.hidden2(x))      # activation function for hidden layer
        #x = F.relu(self.hidden3(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class task1_classifier(object):
    def __init__(self):
        self.net = Net(n_feature=5, n_hidden=50, n_output=3)
        self.net = torch.nn.DataParallel(self.net)
        self.net = self.net.to(device)
        #print(net)  # net architecture
        self.min_max_scaler = MinMaxScaler()

    def scale(self, train_X, test_X):
        #print(train_X.columns)
        #print(test_X.columns)
        #input()
        all_X = pd.concat([train_X, test_X], axis=0,sort=False)
        self.min_max_scaler.fit(all_X)
        return self

    def fit(self, train_X, train_y, epochs=100):
        x = self.min_max_scaler.transform(train_X)
        #x = train_X.to_numpy()
        
        #train_target = torch.tensor(train_y.values.astype(np.float32))
        #train = torch.tensor(train_X.values.astype(np.float32)) 
        #train_tensor = data_utils.TensorDataset(train) 
        #train_loader = data_utils.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = True)
        x = torch.from_numpy(x).float()#.to(device)
        y = torch.from_numpy(train_y.values).float().reshape(-1, 1)#.to(device)
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.02)
        loss_func = torch.nn.CrossEntropyLoss()  # this is for regression mean squared loss
        batch_size = 25600
        train = torch.utils.data.TensorDataset(x, y)
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

        for t in range(epochs):
            for step, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                prediction = self.net(batch_x)     # input x and predict based on x
                #print(y, type(y))
                loss = loss_func(prediction, batch_y.long().squeeze(1))     # must be (1. nn output, 2. target)
                print(t, loss)

                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients

            #if t % 5 == 0:
            #    # plot and show learning process
            #    plt.cla()
            #    plt.scatter(x.data.numpy(), y.data.numpy())
            #    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            #    plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
            #    plt.pause(0.1)
        return self

    def predict(self, test_X):
        x = self.min_max_scaler.transform(test_X)
        #x = test_X.to_numpy()
        x = torch.from_numpy(x).float().to(device)
        prediction = self.net(x)
        #print(prediction)
        return torch.max(prediction, 1)[1].long()

# torch.manual_seed(1)    # reproducible

#x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
#y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()




#plt.ion()   # something about plotting

#plt.ioff()
#plt.show()
