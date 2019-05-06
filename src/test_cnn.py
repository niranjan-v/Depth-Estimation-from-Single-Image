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


Xtest = torch.from_numpy(np.array([[el[:,:,0],el[:,:,1],el[:,:,2]] for el in Xtest.numpy()]))
Ytest = torch.from_numpy(np.array([el[::4,::4].flatten() for el in Ytest.numpy()]))

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

model.load_state_dict(torch.load("models/CnnFCmodel.pt"))
model.float().cuda()

print("Created net")

crit = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4,weight_decay=1e-4)


batchSize = 32
numBatches = int(np.ceil(Xtest.shape[0] / batchSize))

tot_loss = 0
lr=[]
for batchNum in range(numBatches) :
    st0 = time.time()
    XBatch = Xtest[batchNum*batchSize: (batchNum+1)*batchSize].float().cuda()
    YBatch = Ytest[batchNum*batchSize: (batchNum+1)*batchSize].float().cuda()
    output = model(XBatch)
    opt
    loss = crit(output,YBatch)
    lr.append(loss.item())
    tot_loss += loss.item()

print('Test Loss',(1.0*tot_loss)/numBatches)
print(lr.index(min(lr)), min(lr), len(lr))

test = Xtest[100]
ans = Ytest[100]
model.load_state_dict(torch.load("models/CnnFCmodel.pt"))
output = model(test.reshape(1,3,212,280).float().cuda())
# output = output[0].clamp(min=0,max=254)
print(output.min(),output.max())
output = output.reshape([53,70])
img = np.ndarray((212,280,3),dtype='uint8')
test = test.detach().numpy()
img[:,:,0] = test[0]
img[:,:,1] = test[1]
img[:,:,2] = test[2]

f, (ax1,ax2,ax3) = plt.subplots(1,3)
ax1.imshow(img)
ax2.imshow(output.cpu().detach().numpy(),cmap='hot')
ax3.imshow(ans.reshape(53,70),cmap='hot')
plt.show()
