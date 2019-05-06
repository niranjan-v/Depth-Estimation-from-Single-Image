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
import torchvision.models as models
from model_nets import upmodel, fullmodel, finemodel

Xtrain = torch.load("../nyud/images.bin")
Ytrain = torch.load("../nyud/depth.bin")

tot = 32*40
tes = 1449 - tot

Xtest = Xtrain[-tes:]
Ytest = Ytrain[-tes:]

Xtrain = Xtrain[0:tot]
Ytrain = Ytrain[0:tot]

Xtest = torch.from_numpy(np.array([[el[:,:,0],el[:,:,1],el[:,:,2]] for el in Xtest.numpy()]))
Ytest = torch.from_numpy(np.array([el[::4,::4] for el in Ytest.numpy()]))

Xtrain = torch.from_numpy(np.array([[el[:,:,0],el[:,:,1],el[:,:,2]] for el in Xtrain.numpy()]))
Ytrain = torch.from_numpy(np.array([el[::4,::4] for el in Ytrain.numpy()]))

## TEST

model_temp = fullmodel()
model_temp.float().cuda()
model_temp.load_state_dict(torch.load("../bins/residual/BestResidualmodel2.pt"))
crit = nn.MSELoss()

### Display Results

test = Xtest[0:96]
ans = Ytest[0:96]


batchSize = 32
numBatches = int(np.ceil(test.shape[0] / batchSize))

for i in range(numBatches):
  output = model_temp(test[32*(i):32*(i+1)].reshape(batchSize,3,212,280).float().cuda())
  for j in range(batchSize):
    output_j = (output[j]).reshape([53,70])
    img = np.ndarray((212,280,3),dtype='uint8')
    test_i = test[i*32+j].detach().numpy()
    img[:,:,0] = test_i[0]
    img[:,:,1] = test_i[1]
    img[:,:,2] = test_i[2]

    print(32*i+j)
    f, (ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(img)
    ax2.imshow(output_j.cpu().detach().numpy(),cmap='hot')
    ax3.imshow(ans[32*i+j].reshape(53,70),cmap='hot')
    plt.show()