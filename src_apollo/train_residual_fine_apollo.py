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

### UNCOMMENT TO STORE WEIGHTS
# torch.save(torch.FloatTensor([valLoss,trainLoss]),"../bins/residual_fine_apollo/residual_loss_fine_A.bin")

# torch.save(model_fine.state_dict(),"../bins/residual_fine_apollo/BestResidualmodelFine1_A.pt")