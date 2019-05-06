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

XtrainFlip = torch.from_numpy(np.array([[np.fliplr(el[:,:,0]),np.fliplr(el[:,:,1]),np.fliplr(el[:,:,2])] for el in Xtrain.numpy()]))
Xtrain = torch.from_numpy(np.array([[el[:,:,0],el[:,:,1],el[:,:,2]] for el in Xtrain.numpy()]))
YtrainFlip = torch.from_numpy(np.array([np.fliplr(el[::4,::4]) for el in Ytrain.numpy()]))
Ytrain = torch.from_numpy(np.array([el[::4,::4] for el in Ytrain.numpy()]))


model_fine = finemodel()
model_fine.float().cuda()

print("Created net")

crit = nn.MSELoss()
optimizer = torch.optim.Adam(model_fine.parameters(),lr = 1e-2,weight_decay=1e-4)

batchSize = 32
epochs = 30


numBatches = int(np.ceil(Xtrain.shape[0]/ batchSize))
valBatches = int(np.ceil(Xtest.shape[0]/ batchSize))

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
        
      
        
        r1, g1, b1 = random.uniform(0.8,1.2), random.uniform(0.8,1.2), random.uniform(0.8,1.2)
        XBatch[:,0] = XBatch[:,0]*r1
        XBatch[:,1] = XBatch[:,1]*g1
        XBatch[:,2] = XBatch[:,2]*b1

        output = model_fine(XBatch)
        # print(output)
        optimizer.zero_grad()
        loss = crit(output,YBatch)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        
        XBatch = XtrainFlip[batchNum*batchSize: (batchNum+1)*batchSize].float().cuda()
        YBatch = YtrainFlip[batchNum*batchSize: (batchNum+1)*batchSize].float().cuda()
        
        r1, g1, b1 = random.uniform(0.8,1.2), random.uniform(0.8,1.2), random.uniform(0.8,1.2)
        XBatch[:,0] = XBatch[:,0]*r1
        XBatch[:,1] = XBatch[:,1]*g1
        XBatch[:,2] = XBatch[:,2]*b1
        
        output = model_fine(XBatch)
        optimizer.zero_grad()
        loss = crit(output,YBatch)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        # print("BatchNum",batchNum,time.time()-st0)
    tot_loss = tot_loss/2
    
    val_loss = 0
    for batchNum in range(valBatches) :
        st0 = time.time()
        XBatch = Xtest[batchNum*batchSize: (batchNum+1)*batchSize].float().cuda()
        YBatch = Ytest[batchNum*batchSize: (batchNum+1)*batchSize].float().cuda()

        output = model_fine(XBatch)
        loss = crit(output,YBatch)
        val_loss += loss.item()
        # print("BatchNum",batchNum,time.time()-st0)

    print('epoch :',ep,' time :',time.time()-st)
    print('Training Loss',(1.0*tot_loss)/numBatches)
    trainLoss.append((1.0*tot_loss)/numBatches)
    print('Validation Loss',(1.0*val_loss)/valBatches)
    valLoss.append((1.0*val_loss)/valBatches)

## UNCOMMENT THIS TO SAVE MODELS
# torch.save(torch.FloatTensor([valLoss,trainLoss]),"../bins/residual_fine/residual_loss_fine.bin")

# torch.save(model_fine.state_dict(),"../bins/residual_fine/BestResidualmodelFine1.pt")