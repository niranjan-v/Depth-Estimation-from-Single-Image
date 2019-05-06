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

Xtrain = torch.load("../nyud/images.bin")
Ytrain = torch.load("../nyud/depth.bin")

tot = 32*40
tes = 1449 - tot

Xtest = Xtrain[-tes:]
Ytest = Ytrain[-tes:]

Xtrain = Xtrain[0:tot]
Ytrain = Ytrain[0:tot]

Xtest = torch.from_numpy(np.array([[el[:,:,0],el[:,:,1],el[:,:,2]] for el in Xtest.numpy()]))
Ytest = torch.from_numpy(np.array([el[::4,::4].flatten() for el in Ytest.numpy()]))

Xtrain = torch.from_numpy(np.array([[el[:,:,0],el[:,:,1],el[:,:,2]] for el in Xtrain.numpy()]))
Ytrain = torch.from_numpy(np.array([el[::4,::4].flatten() for el in Ytrain.numpy()]))

def weights_init(m):
    if type(m) in [nn.Linear]:
        print("setting custom wts")
        #m.weight.data.register_hook(lambda grad: print(grad))
        m.weight.data = torch.randn(m.weight.data.shape).float() * math.sqrt(2/m.weight.data.shape[1])
        m.bias.data = torch.randn(m.bias.data.shape).float() * math.sqrt(2/m.weight.data.shape[1])
        #print(m.weight.data, m.bias.data)
    elif type(m) in [nn.Conv2d]:
        print("setting custom wts")
        #m.weight.data.register_hook(lambda grad: print(grad))
        m.weight.data = torch.randn(m.weight.data.shape).float() * math.sqrt(2/(m.weight.data.shape[1]*m.weight.data.shape[2]*m.weight.data.shape[3]))
        m.bias.data = torch.randn(m.bias.data.shape).float() * math.sqrt(2/(m.weight.data.shape[1]*m.weight.data.shape[2]*m.weight.data.shape[3]))
        #print(m.weight.data, m.bias.data)



## TEST
model_temp = nn.Sequential(
    nn.Conv2d(in_channels=3,out_channels=32,kernel_size=11,stride=1,padding=5),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.Conv2d(in_channels=64,out_channels=32,kernel_size=5,stride=1,padding=2),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(16),
    nn.MaxPool2d(kernel_size=2),

    Flatten(),

    nn.Linear(in_features=53*70*16,out_features=3710),
    nn.ReLU(),
    nn.Linear(in_features=3710,out_features=3710),
    nn.ReLU()

    )
model_temp.float().cuda()
model_temp.load_state_dict(torch.load("../bins/CNN+FC_med/CnnFCmodel.pt"))
crit = nn.MSELoss()


batchSize = 32
numBatches = int(np.ceil(Xtest.shape[0] / batchSize))
tot_loss = 0
for batchNum in range(numBatches) :
    st0 = time.time()
    XBatch = Xtest[batchNum*batchSize: (batchNum+1)*batchSize].float().cuda()
    YBatch = Ytest[batchNum*batchSize: (batchNum+1)*batchSize].float().cuda()

    output = model_temp(XBatch)
    # print(output.shape)
    loss = crit(output,YBatch)
    # output.register_hook(lambda grad : print(grad))
    tot_loss += loss.item()
    # print("BatchNum",batchNum,time.time()-st0)

print('Test Loss',(1.0*tot_loss)/numBatches)


## DISPLAY SOME RESULTS

test = Xtest[0:96]
ans = Ytest[0:96]

crit = nn.MSELoss()

for i in range(numBatches):
  output = model_temp(test[batchSize*(i):batchSize*(i+1)].reshape(batchSize,3,212,280).float().cuda())
  loss = crit(output,YBatch)
  for j in range(batchSize):
    output_j = (output[j]).reshape([53,70])
    img = np.ndarray((212,280,3),dtype='uint8')
    test_i = test[i*batchSize+j].detach().numpy()
    img[:,:,0] = test_i[0]
    img[:,:,1] = test_i[1]
    img[:,:,2] = test_i[2]

    print(batchSize*i+j)
    print("Loss ",loss)
    f, (ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(img)
    ax2.imshow(output_j.cpu().detach().numpy(),cmap='hot')
    ax3.imshow(ans[batchSize*i+j].reshape(53,70),cmap='hot')
    plt.show()