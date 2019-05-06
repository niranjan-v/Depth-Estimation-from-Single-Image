import torch
import imageio
import torch.nn as nn
import glob
import time
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import math
import random

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

Xtrain = torch.load("data/images.bin")
Ytrain = torch.load("data/depth.bin")

tot = 32*40
tes = 1449 - tot

Xtest = Xtrain[-tes:]
Ytest = Ytrain[-tes:]

Xtrain = Xtrain[0:tot]
Ytrain = Ytrain[0:tot]

Xtest = torch.from_numpy(np.array([[el[:,:,0],el[:,:,1],el[:,:,2]] for el in Xtest.numpy()]))
Ytest = torch.from_numpy(np.array([el[::4,::4].flatten() for el in Ytest.numpy()]))

XtrainFlip = torch.from_numpy(np.array([[np.fliplr(el[:,:,0]),np.fliplr(el[:,:,1]),np.fliplr(el[:,:,2])] for el in Xtrain.numpy()]))
Xtrain = torch.from_numpy(np.array([[el[:,:,0],el[:,:,1],el[:,:,2]] for el in Xtrain.numpy()]))
YtrainFlip = torch.from_numpy(np.array([np.fliplr(el[::4,::4]).flatten() for el in Ytrain.numpy()]))
Ytrain = torch.from_numpy(np.array([el[::4,::4].flatten() for el in Ytrain.numpy()]))

def weights_init(m):
    if type(m) in [nn.Linear]:
        print("setting custom wts")
        m.weight.data = torch.randn(m.weight.data.shape).float() * math.sqrt(2/m.weight.data.shape[1])
        m.bias.data = torch.randn(m.bias.data.shape).float() * math.sqrt(2/m.weight.data.shape[1])
    elif type(m) in [nn.Conv2d]:
        print("setting custom wts")
        m.weight.data = torch.randn(m.weight.data.shape).float() * math.sqrt(2/(m.weight.data.shape[1]*m.weight.data.shape[2]*m.weight.data.shape[3]))
        m.bias.data = torch.randn(m.bias.data.shape).float() * math.sqrt(2/(m.weight.data.shape[1]*m.weight.data.shape[2]*m.weight.data.shape[3]))

model = nn.Sequential(
    nn.Conv2d(in_channels=3,out_channels=50,kernel_size=3,stride=1,padding=1), # Size = 212 x 280 x 50
    nn.ReLU(),
    nn.BatchNorm2d(50),
    #
    nn.Conv2d(in_channels=50,out_channels=50,kernel_size=3,stride=1,padding=1), # Size = 212 x 280 x 50
    nn.ReLU(),
    nn.BatchNorm2d(50),
    #
    nn.Conv2d(in_channels=50,out_channels=50,kernel_size=3,stride=1,padding=1), # Size = 212 x 280 x 50
    nn.ReLU(),
    nn.BatchNorm2d(50),
    #
    nn.Conv2d(in_channels=50,out_channels=80,kernel_size=3,stride=1,padding=1), # op: Size = 106 x 140 x 80
    nn.ReLU(),
    nn.BatchNorm2d(80),
    nn.MaxPool2d(kernel_size=2),
    ##
    nn.Conv2d(in_channels=80,out_channels=100,kernel_size=3,stride=1,padding=1), 
    nn.ReLU(),
    nn.BatchNorm2d(100),
    nn.MaxPool2d(kernel_size=2), #op Size = 53 x 70 x 80
    ##
    nn.Conv2d(in_channels=100,out_channels=120,kernel_size=3,stride=1,padding=1), 
    nn.ReLU(),
    nn.BatchNorm2d(120),
    nn.MaxPool2d(kernel_size=2), # Size = 26 x 35 x 80
    ##
    nn.Conv2d(in_channels=120,out_channels=120,kernel_size=3,stride=1,padding=1),#26*35
    nn.ReLU(),
    nn.BatchNorm2d(120),
    nn.ConvTranspose2d(in_channels=120,out_channels=120,kernel_size=[3,2],stride=2,padding=0), # Size = 53 x 70 x 120
    ##
    nn.Conv2d(in_channels=120,out_channels=1,kernel_size=3,stride=1,padding=1), # Size = 53 x 70 x 1
    nn.ReLU(),
    Flatten()

    )
#model.apply(weights_init)
#model.load_state_dict(torch.load("data/CnnFCmodel.pt"))
model.float().cuda()

print("Created net")

crit = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4,weight_decay=1e-4)

batchSize = 16
epochs = 30


numBatches = int(np.ceil(Xtrain.shape[0] / batchSize))
valBatches = int(np.ceil(Xtest.shape[0] / batchSize))

valLoss = []
trainLoss = []

for ep in range(epochs) :
    perm = torch.randperm(Xtrain.shape[0])
    Xtrain=Xtrain[perm]
    XtrainFlip = XtrainFlip[perm]
    Ytrain=Ytrain[perm]
    YtrainFlip = YtrainFlip[perm]

    st = time.time()
    print('start epoch',ep)
    
    tot_loss = 0  
    for batchNum in range(numBatches) :
        st0 = time.time()
        XBatch = Xtrain[batchNum*batchSize: (batchNum+1)*batchSize].float().cuda()
        YBatch = Ytrain[batchNum*batchSize: (batchNum+1)*batchSize].float().cuda()

        output = model(XBatch)
        optimizer.zero_grad()
        loss = crit(output,YBatch)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

        
        XBatch = XtrainFlip[batchNum*batchSize: (batchNum+1)*batchSize].float().cuda()
        YBatch = YtrainFlip[batchNum*batchSize: (batchNum+1)*batchSize].float().cuda()
        
        output = model(XBatch)
        optimizer.zero_grad()
        loss = crit(output,YBatch)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    tot_loss = tot_loss/2
    val_loss = 0
    for batchNum in range(valBatches) :
        st0 = time.time()
        XBatch = Xtest[batchNum*batchSize: (batchNum+1)*batchSize].float().cuda()
        YBatch = Ytest[batchNum*batchSize: (batchNum+1)*batchSize].float().cuda()

        output = model(XBatch)
        loss = crit(output,YBatch)
        val_loss += loss.item()
    print('epoch :',ep,' time :',time.time()-st)
    print('Training Loss',(1.0*tot_loss)/numBatches)
    trainLoss.append((1.0*tot_loss)/numBatches)
    print('Validation Loss',(1.0*val_loss)/valBatches)
    valLoss.append((1.0*val_loss)/valBatches)

    
torch.save(model.state_dict(),"models/CnnFCmodel.pt")
torch.save(torch.FloatTensor([valLoss,trainLoss]),"models/loss.bin")
