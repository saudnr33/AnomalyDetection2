import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from ucsd_dataset import UCSDAnomalyDataset
from video_CAE import VideoAutoencoderLSTM
import torch.backends.cudnn as cudnn
import numpy as np
#matplotlib notebook
import matplotlib.pyplot as plt
from Labels import Labels
from Reader import Reader

from Autoencoder import NeuralNet

net = torch.load("TheAuto6")

file = Reader("UCSDped1/TestModified/")
dictt = file.getFrames()
keys = dictt.keys()

file = Reader("UCSDped1/Train/")
dictt2 = file.getFrames()
keys2 = dictt.keys()


for key in keys:
    Temp = []
    array = dictt["Test027"]
    aa = dictt2["Train001"]
    for i in range(0, len(array) - 5, 5):
        x  = (torch.FloatTensor(array[50:55])/255 - 0.5)

        x2 =  (torch.FloatTensor(aa[50:55])/255 - 0.5)

        testSet = x.view(5, 1, 158, -1)
        test2 =  x2.view(5, 1, 158, -1)
        y = net.forward(testSet.cuda())
        y2 = net.forward(test2.cuda())
        break
    break


yy2 = y2.cpu().detach().numpy()
yy = y.cpu().detach().numpy()
xx = x[0].detach().numpy()
yy = yy[0][0]
xx = (xx + 0.5) * 255
yy = (yy + 0.5) * 255

yy2 = (yy2[0][0] + 0.5) * 255

plt.imshow(yy2, cmap="gray")
plt.show()

plt.imshow(xx, cmap="gray")
plt.show()


plt.imshow(yy, cmap="gray")
plt.show()
