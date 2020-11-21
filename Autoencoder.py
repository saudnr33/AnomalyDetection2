import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Reader import Reader
from Labels import Labels
from os import listdir
from tifffile import imsave
import matplotlib.pyplot as plt
from convolution_lstm import ConvLSTM


z =  ConvLSTM(input_channels=64, hidden_channels=[64, 32, 64], kernel_size=3, batch_first=True, input_dropout_rate=0.5, reccurent_dropout_rate=0.5)
class NeuralNet(torch.nn.Module):
	def __init__(self, lrate, loss_fn):
		super(NeuralNet, self).__init__()

		self.latent = []

		self.conv1 = nn.Conv2d(1, 128, 3, stride=1, padding=1)
		self.maxPool = nn.MaxPool2d(2, 2)
		self.maxPool2 = nn.MaxPool2d(2, 2)
		self.SELU = nn.SELU()
		self.conv2 = nn.Conv2d(128,64, 3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.rnn_enc_dec = ConvLSTM(input_channels=64, hidden_channels=[64, 32, 64], kernel_size=3, batch_first=True, input_dropout_rate=0.5, reccurent_dropout_rate=0.5)
		# self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

		self.convTrans3 = nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1)
		self.upSample3 =  nn.Upsample((20, 30))
		self.convTrans1 = nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1)
		self.upSample1 =  nn.Upsample((158, 238))
		self.upSample2 =  nn.Upsample((158, 238))
		self.upSample3 =  nn.Upsample((79, 119))
		self.convTrans2 = nn.ConvTranspose2d(128, 1, 3, stride=1, padding=1)
		self.linear = nn.Linear(19*29, 19*29)
		self.linear2 = nn.Linear(19*29, 19*29)
		# self.encoder = nn.Sequential(
		# 	nn.Conv2d(1, 16, 3, stride=2, padding=1),
		# 	nn.MaxPool2d(2, 2),
		# 	nn.SELU(),
		# 	nn.Conv2d(16, 32, 3, stride=2, padding=1),
		# 	nn.MaxPool2d(2, 2),
		# 	nn.SELU(),
		# )
		# self.decoder = nn.Sequential(
		# 	nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),
		# 	nn.MaxUnpool2d(2, stride = 2),
		# 	nn.SELU(),
		# 	nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
		# 	nn.MaxUnpool2d(2, stride = 2),
		# 	nn.SELU(),
		# )

		# self.encoded_layer1 = nn.Conv2d(1, 64, 3)
		# self.encoded_layer11 = nn.MaxPool2d( (2, 2))
		# self.encoded_layer2 = nn.Conv2d(64, 32, 3)
		# self.encoded_layer22 = nn.MaxPool2d( (2, 2))
		# self.encoded_layer3 = nn.Conv2d(32, 16, 3)
		# self.latent_view   = nn.MaxPool2d( (2, 2))
		#
		# # decoding architecture
		# self.decoded_layer1 = nn.Conv2d(16, 16, 3)
		# self.decoded_layer11 = nn.Upsample((38, 58))
		# self.decoded_layer2 = nn.Conv2d(16, 32, 3)
		# self.decoded_layer22 = nn.Upsample((76, 116))
		# self.decoded_layer3 = nn.Conv2d(32, 64, 3)
		# self.decoded_layer33 = nn.Upsample((156, 236))
		# self.output_layer   = nn.Conv2d(64, 1,  3, padding = 2)

		self.loss_fn = loss_fn
		self.lrate = lrate
		self.sigmoid =  nn.Sigmoid()
	def get_parameters(self):
		return self.parameters()

	def forward(self, x):
		batchSize = x.size()[0]
		x = x.view(5,1, 158, -1)
		# x = self.encoded_layer1(x)
		# x = F.relu(x)
		# x = self.encoded_layer11(x)
		# x = self.encoded_layer2(x)
		# x = F.relu(x)
		# x = self.encoded_layer22(x)
		# x = self.encoded_layer3(x)
		# x = F.relu(x)
		# x = self.latent_view(x)
		#
		# x = self.decoded_layer1(x)
		# x = F.relu(x)
		# x = self.decoded_layer11(x)
		# x = self.decoded_layer2(x)
		# x = F.relu(x)
		# x = self.decoded_layer22(x)
		# x = self.decoded_layer3(x)
		# x = F.relu(x)
		# x = self.decoded_layer33(x)
		# x = self.output_layer(x)
		# x = self.sigmoid(x)


		# print(x.size())
		x = self.conv1(x)
		# print(x.size() , "After Conv1")
		x =  self.tanh(x)
		# x = self.maxPool(x)
		# print(x.size() , "After mapool1")
		x = self.conv2(x)
		# print(x.size() , "After Conv2")
		x =  self.tanh(x)
		# x = self.maxPool2(x)
		# print(x.size() , "After maxPool2")
		# x = self.conv3(x)
		# # print(x.size() , "After Conv2")
		# x =  self.tanh(x)
		# x = self.maxPool(x)
		# ''' '''
		# x = x.view(1,128, 19*29)
		# x = self.linear(x)
		# x = self.linear2(x)
		# x = x.view(1, 128, 19, 29)
		''' '''
		x = x.view(1, 5, 64, x.size()[-2], x.size()[-1])
		x, _ = self.rnn_enc_dec(x)
		x = x.view(5, 64,  x.size()[-2], x.size()[-1])
		# if state == True:
		# 	self.latent.append(x)
		# x = self.convTrans3(x)
		# # print(x.size(), " After conv3")
		#
		# x =  self.tanh(x)
		# x = self.upSample3(x)
		# print(x.size(), " After smaple3")
		x = self.convTrans1(x)
		# print(x.size(), "After cobnvT1")

		x =  self.tanh(x)
		# x = self.upSample1(x)
		# print(x.size(), "After sample1")
		x = self.convTrans2(x)
		# print(x.size(), "After cobnvT2")
		x =  self.tanh(x)
		# x = self.upSample2(x)
		# print(x.size(), "After sample1")
		# x =  self.tanh(x)
		# x = self.sigmoid(x)
		# x = self.decoder(x)
		return x

	def step(self, x, x_toCompare):
		optimizer = optim.Adam(self.parameters(), lr=1e-4, eps=1e-6, weight_decay=1e-5)
		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = self(x)
		loss = self.loss_fn(outputs, x_toCompare)
		loss.backward()
		optimizer.step()
		return loss.item()
	def getLatent():
		if len(self.latent) ==0:
			return "Latent has not been initialized"
		else:
			return self.latent

device = torch.device("cuda:0")

file = Reader("UCSDped1/Train/")
data_dictionary = file.getFrames()
loss_function = torch.nn.MSELoss()
net = NeuralNet(0.0001, loss_function).to(device)


from pytorch_model_summary import summary
print(summary(net, torch.zeros((1, 5, 158, 238)).cuda(), show_input=True))

def fit(data, net):
	keys = data.keys()
	print(keys)
	for epoch in range(1):
		print("EPoch = " , epoch)
		for key in keys:
			print(" key =================== ", key)
			array = data[key]
			for i in range(len(array) - 30):
				x = torch.FloatTensor(array[i:i+5])
				trainingSet =  (x/255 - 0.5)

				xxz = torch.FloatTensor(array[i+25:i+30]).view(5, 1, 158, -1)
				xxz = (xxz/255 - 0.5)
				x_toCompare = xxz.cuda()
				loss = net.step(trainingSet.view(5,1, 158, -1).cuda(), x_toCompare)
				print("Loss = ", loss)
#
# print("FIT")
# fit(data_dictionary, net)
# print("WORKS")
#
# torch.save(net, "TheAuto6")

#
#
# "Inf"
# print("-------------------------------------------------------")
#
# """ Extract labels """
#
#
#
# file = Reader("UCSDped1/TestModified/")
# dictt = file.getFrames()
# keys = dictt.keys()
# lossArr = []
# minArr = []
# maxArr = []
#
# thre = 1
#
# c = 0
#
# AllLabels = Labels().labels(1)
#
# COUNT_MAJOR = []
# for key in keys:
# 	if c == 10:
# 		break
# 	Temp = []
# 	array = dictt[key]
# 	print(len(array) - 10)
#
# 	for i in range(0, len(array) - 5, 5):
# 		x  = (torch.FloatTensor(array[i:i+5])/255 - 0.5)
# 		testSet = x.view(5, 1, 158, -1)
# 		y = net.forward(testSet.cuda())
#
#
# 		xx = (torch.FloatTensor(array[i:i+5])/255 - 0.5).cuda().view(5, 1, 158, -1)
# 		# print(y.size(), testSet.size())
#
# 		loss = loss_function(y, xx)
#
# 		# loss = torch.norm(xx.cpu().data.view(xx.size(0),-1) - y.cpu().data.view(y.size(0),-1), dim=1)
# 		print("Loss = ", loss.item(), "; key = ", key, "; frame = ", i+5)
# 		Temp.append(loss.item())
#
#
# 	c+=1
# 	lossArr.append(Temp)
# 	minArr.append(np.min(Temp))
# 	maxArr.append(np.max(Temp))
#
#
#
# maxx = np.max(maxArr)
# minn = np.min(minArr)
# score_Matrix = lossArr
#
# thesholds = np.linspace(0, 1, 100)
#
#
# ##We check labels
# Accuaracies = np.zeros(len(thesholds))
# for i in range(len(score_Matrix)):
# 	size = len(thesholds)
# 	correctAnom = np.zeros(size)
# 	correctNorm = np.zeros(size)
# 	numofZeros = 0
# 	numofOnes = 0
# 	z = -1
# 	for j in range(0, len(array) - 5, 5):
# 		z+=1
# 		label =int(sum( AllLabels[i][j:j + 5])/3)
# 		print(i, j, c, len(score_Matrix[i]))
# 		score = (score_Matrix[i][z] - min(score_Matrix[i])) / (max(score_Matrix[i]) - min(score_Matrix[i]))
# 		if label == 1:
# 			numofOnes +=1
# 		else:
# 			numofZeros +=1
# 		for c in range(size):
# 			thre = thesholds[c]
#
# 			if score > thre:
# 				if label == 1:
# 					correctAnom[c] +=1
# 			if score <= thre:
# 				if label == 0:
# 					correctNorm[c] +=1
# 	acc = (correctAnom + correctNorm)/(numofOnes + numofZeros)
# 	anonlabelcorrect = (correctAnom)/(numofOnes + 1)
# 	normlabelcorrect = (correctNorm)/(numofZeros + 1)
# 	Accuaracies += acc
# 	print("Acc = ", acc, "  NormCorrect = ", normlabelcorrect, "  AnonCorrect = ", anonlabelcorrect)
#
# print("-------------------------")
# print(Accuaracies/(len(score_Matrix)))
#
#
# a = (lossArr[0] - min(lossArr[0]))/ (max(lossArr[0] - min(lossArr[0])) )
# # b = (lossArr[4] - minn) / (maxx)
# # c = (lossArr[6] - minn) / (maxx)
# # d  = (lossArr[13] - minn) / (maxx - minn)
# plt.plot(a)
# plt.show()
# # plt.plot(b)
# # plt.show()
# # plt.plot(c)
# # plt.show()
# plt.plot(thesholds, Accuaracies/(len(score_Matrix)))
# plt.show()
# # from PIL import Image
# # im = Image.fromarray(ans[0].detach().numpy())
# # im.save("BS/001.tif")
# #
