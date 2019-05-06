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
from model_nets_apollo import upmodel, finemodel, fullmodel

Xtrain = torch.load("../apolloscape/apollo_images.bin")
Ytrain = torch.load("../apolloscape/apollo_depth.bin")

tot = 32*50
tes = 2000 - tot

Xtest = Xtrain[-tes:]
Ytest = Ytrain[-tes:]

Xtrain = Xtrain[0:tot]
Ytrain = Ytrain[0:tot]

Xtest = torch.from_numpy(np.array([[el[:,:,0],el[:,:,1],el[:,:,2]] for el in Xtest]))
Ytest = torch.from_numpy(np.array([el[::4,::4]/10.0 for el in Ytest]))

XtrainFlip = torch.from_numpy(np.array([[np.fliplr(el[:,:,0]),np.fliplr(el[:,:,1]),np.fliplr(el[:,:,2])] for el in Xtrain]))
Xtrain = torch.from_numpy(np.array([[el[:,:,0],el[:,:,1],el[:,:,2]] for el in Xtrain]))
YtrainFlip = torch.from_numpy(np.array([np.fliplr(el[::4,::4]/10.0) for el in Ytrain]))
Ytrain = torch.from_numpy(np.array([np.uint8(el[::4,::4]/10.0) for el in Ytrain]))


## TEST

model_temp = fullmodel()
model_temp.float().cuda()
model_temp.load_state_dict(torch.load("../bins/residual_apollo/BestResidualmodel1_A.pt"))
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

## DISPLAY

test = Xtest[96:96*2]
ans = Ytest[96:96*2]

for i in range(numBatches):
  output = model_temp(test[batchSize*(i):batchSize*(i+1)].reshape(batchSize,3,226,282).float().cuda())
  for j in range(batchSize):
    output_j = (output[j]).reshape([57,71])
    img = np.ndarray((226,282,3),dtype='uint8')
    test_i = test[i*batchSize+j].detach().numpy()
    img[:,:,0] = test_i[0]
    img[:,:,1] = test_i[1]
    img[:,:,2] = test_i[2]

    print(batchSize*i+j)
    f, (ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(img)
    ax2.imshow(output_j.cpu().detach().numpy(),cmap='hot')
    ax3.imshow(ans[batchSize*i+j].reshape(57,71),cmap='hot')
    plt.show()