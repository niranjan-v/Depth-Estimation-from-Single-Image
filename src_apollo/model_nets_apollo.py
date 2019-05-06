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



class upmodel(nn.Module):
  def __init__(self,inp_dim):
    super(upmodel,self).__init__()
    self.inp_dim = inp_dim
    self.unpool = nn.MaxUnpool2d(kernel_size=2,stride=2)
    self.conv1 = nn.Conv2d(in_channels=inp_dim[0],out_channels=int(inp_dim[0]/2),kernel_size=5,stride=1,padding=2)
    self.relu = nn.ReLU()
    self.conv2 = nn.Conv2d(in_channels=int(inp_dim[0]/2),out_channels=int(inp_dim[0]/4),kernel_size=3,stride=1,padding=1)
    self.conv_side = nn.Conv2d(in_channels=int(inp_dim[0]),out_channels=int(inp_dim[0]/4),kernel_size=5,stride=1,padding=2)
  def forward(self,inp):
    x = torch.zeros(inp.shape[0],inp.shape[1],2*inp.shape[2],2*inp.shape[3]).float().cuda()
    x[:,:,::2,::2] = inp
    #print(x)
    # Without using maxUnpool3d
    x1 = self.conv1(x)
    x1 = self.relu(x1)
    x1 = self.conv2(x1)
    x2 = self.conv_side(x)
    return torch.cat((x1,x2),dim = 1)
class fullmodel(nn.Module):
  def __init__(self):
    super(fullmodel,self).__init__()
    self.res = models.resnet50(pretrained=True)
    self.res = nn.Sequential(*(list(self.res.children())[:-2]))
    for param in self.res.parameters():
      param.requires_grad = False
    self.first_conv = nn.Conv2d(in_channels=2048,out_channels=1024,kernel_size=1,stride=1,padding=0)
    self.batchNorm = nn.BatchNorm2d(1024)
    self.upsample1 = upmodel([1024,8,9])
    self.upsample2 = upmodel([512,15,18])
    self.upsample3 = upmodel([256,29,36])
    self.convlayer = nn.Conv2d(in_channels=128,out_channels=1,kernel_size=3,stride=1,padding=1)
    self.relu = nn.ReLU()
    self.init_weights()
  def init_weights(self) :
    self.upsample1.apply(weights_init)
    self.upsample2.apply(weights_init)
    self.upsample3.apply(weights_init)
    self.first_conv.apply(weights_init)
    self.convlayer.apply(weights_init)
  def forward(self,inp):
    x = self.res(inp)
    x = self.first_conv(x)
    x = self.batchNorm(x)
    x = self.upsample1(x)
    x = x[:,:,:-1,:]
    x = self.upsample2(x)
    x = x[:,:,:-1,:]
    x = self.upsample3(x)
    x = self.convlayer(x)
    x = self.relu(x)
    x = x[:,:,:-1,:-1]
    x1 = torch.zeros(x.shape[0],x.shape[2],x.shape[3]).cuda()
    for i in range(x.shape[0]):
      x1[i] = x[i,0,:,:]
    return x1


class finemodel(nn.Module):
  def __init__(self):
    super(finemodel,self).__init__()
    self.coarse_model = fullmodel()
    self.coarse_model.float().cuda()
    self.coarse_model.load_state_dict(torch.load("../bins/residual_apollo/BestResidualmodel1_A.pt"))
    for param in self.coarse_model.parameters():
      param.requires_grad = False
    self.conv_start = nn.Conv2d(in_channels=3,out_channels=63,kernel_size=9,stride=2,padding=5)
    self.relu_start = nn.ReLU()
    self.batch_start = nn.BatchNorm2d(63)
    self.pool = nn.MaxPool2d(kernel_size = 2)
    self.conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,stride=1,padding=2)
    self.relu1 = nn.ReLU()
    self.batch1 = nn.BatchNorm2d(64)
    self.conv2 = nn.Conv2d(in_channels=64,out_channels=1,kernel_size=5,stride=1,padding=2)
    self.relu2 = nn.ReLU()
    self.batch2 = nn.BatchNorm2d(1)
    self.init_weights()
  def init_weights(self) :
    self.conv_start.apply(weights_init)
    self.conv1.apply(weights_init)
    self.conv2.apply(weights_init)
  def forward(self,inp) :
    depth = self.coarse_model(inp)
    x1 = torch.zeros(depth.shape[0],1,depth.shape[1],depth.shape[2]).cuda()
    for i in range(depth.shape[0]):
      x1[i,0,:,:] = depth[i,:,:]
    x = self.conv_start(inp)
    x = self.relu_start(x)
    x = self.batch_start(x)
    x = self.pool(x)
    
    x = torch.cat((x,x1),dim = 1)
    
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.batch1(x)
    
    x = self.conv2(x)
    x = self.relu2(x)
    x = self.batch2(x)
    
    x1 = torch.zeros(x.shape[0],x.shape[2],x.shape[3]).cuda()
    for i in range(x.shape[0]):
      x1[i] = x[i,0,:,:]
    return x1
