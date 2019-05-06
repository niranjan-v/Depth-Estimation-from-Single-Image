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


# TEST fine model

model_fine_temp = finemodel()
model_fine_temp.float().cuda()
model_fine_temp.load_state_dict(torch.load("../bins/residual_fine/BestResidualmodelFine1.pt"))
crit = nn.MSELoss()


batchSize = 32
numBatches = int(np.ceil(Xtest.shape[0] / batchSize))
tot_loss = 0
for batchNum in range(numBatches) :
    st0 = time.time()
    XBatch = Xtest[batchNum*batchSize: (batchNum+1)*batchSize].float().cuda()
    YBatch = Ytest[batchNum*batchSize: (batchNum+1)*batchSize].float().cuda()

    output = model_fine_temp(XBatch)
    # print(output.shape)
    loss = crit(output,YBatch)
    # output.register_hook(lambda grad : print(grad))
    tot_loss += loss.item()
    # print("BatchNum",batchNum,time.time()-st0)

print('Test Loss',(1.0*tot_loss)/numBatches)


### Display results
model_temp = fullmodel()
model_temp.float().cuda()
model_temp.load_state_dict(torch.load("../bins/residual/BestResidualmodel2.pt"))

test = Xtest[0:96]
ans = Ytest[0:96]


batchSize = 32
numBatches = int(np.ceil(test.shape[0] / batchSize))

for i in range(numBatches):
  output = model_fine_temp(test[batchSize*(i):batchSize*(i+1)].reshape(batchSize,3,212,280).float().cuda())
  output_2 = model_temp(test[batchSize*(i):batchSize*(i+1)].reshape(batchSize,3,212,280).float().cuda())
  for j in range(batchSize):
    output_j = (output[j]).reshape([53,70])
    output_2_j = (output_2[j]).reshape([53,70])
    img = np.ndarray((212,280,3),dtype='uint8')
    test_i = test[i*batchSize+j].detach().numpy()
    img[:,:,0] = test_i[0]
    img[:,:,1] = test_i[1]
    img[:,:,2] = test_i[2]

    print(batchSize*i+j)
    f, (ax1,ax2,ax3,ax4) = plt.subplots(1,4)
    ax1.imshow(img)
    ax2.imshow(output_j.cpu().detach().numpy(),cmap='hot')
    ax3.imshow(output_2_j.cpu().detach().numpy(),cmap='hot')
    ax4.imshow(ans[batchSize*i+j].reshape(53,70),cmap='hot')
    plt.show()